import torch
from torch.utils.data import Dataset
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import ToktokTokenizer
import math
import random
import os
from collections import Counter, deque
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from threading import Thread
from queue import Empty
from datetime import datetime
from functools import partial
import signal
import string
import linecache

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(0)
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class SingleFileDataset(Dataset):
    """Dataset to prepare a single file of Skip-Gram entries"""

    def __init__(self, path, char_to_index, unigram_table, neg_samples=5):
        self.path = path
        self.c_to_index = char_to_index
        self.table = unigram_table
        self.neg_samples = neg_samples
        self.len = self.file_len(path)

    def __getitem__(self, idx):
        try:
            line = linecache.getline(self.path, idx + 1)
            word, target = line.split('\t')
        except Exception:
            print('Error at line {}'.format(idx))
            print(line)
        negs = self.get_negatives(word)
        negs = [self.prepare_tensor(neg) for neg in negs]
        example = [self.prepare_tensor(word), self.prepare_tensor(target)]
        example += negs
        return example

    def file_len(self, fname):
        with open(fname, 'r', encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        return i - 1

    def __len__(self):
        return self.len - 1

    def prepare_tensor(self, word):
        return prepare_tensor(word, self.c_to_index)

    def get_negatives(self, word):
        to_return = []
        while len(to_return) < self.neg_samples:
            neg = random.choice(self.table)
            if neg == word.lower():
                continue
            to_return.append(neg)
        return to_return


class DataLoaderMultiFiles(object):
    """DataLoader to iterator over a set of DataSet"""

    def __init__(self, dataset, batch_s):
        self.dataset = dataset
        self.batch_size = batch_s
        self.index_queue = deque(torch.randperm(len(self.dataset)).tolist())
        self.batch_queue = Queue(maxsize=5)

    def __iter__(self):
        print('new iteration of dataloader')
        args = (self.batch_queue, self.index_queue, self.dataset,
                self.batch_size)
        self.batch_process = Process(target=fill_batch, args=args)
        self.batch_process.daemon = True
        self.batch_process.start()
        return self

    def is_alive(self):
        # return sum([e.is_alive() for e in self.buffr_processes])
        return self.batch_process.is_alive()

    def __next__(self):
        # print('batch_queue: {}'.format(self.batch_queue.qsize()))
        timeout = 600 if self.is_alive() else 1
        try:
            batch = self.batch_queue.get(timeout=timeout)
        except Empty:
            print('empty')
            self.kill()
            raise StopIteration
        # print('got batch')
        tmp = LongTensor(batch)
        # print('computing')
        return tmp

    def kill(self):
        print('Killing processes')
        self.batch_process.terminate()

    def __del__(self):
        self.kill()


def fill_batch(batch_queue, index_queue, dataset, batch_size):
    batch = []
    counter = 0
    print('filling batch')
    while len(index_queue) > 0:
        index = index_queue.pop()
        example = dataset[index]
        batch.extend(example)
        counter += 1
        if counter == batch_size:
            counter = 0
            batch_queue.put(pad_sequences(batch))
            batch = []
    print('dataset done')


def pad_sequences(sequences):
    max_len = 0
    for sequence in sequences:
        if sequence.shape[0] > max_len:
            max_len = sequence.shape[0]
    padded_sequences = np.zeros([len(sequences), max_len])
    for i, sequence in enumerate(sequences):
        padded_sequences[i, :sequence.shape[0]] = sequence
    return padded_sequences


def prepare_tensor(word, char_to_index):
    unk = char_to_index['UNK']
    indices = [char_to_index['{']]
    for char in word:
        indices.append(char_to_index.get(char, unk))
    indices.append(char_to_index['}'])
    return np.array(indices)


def build_vocab_multi(directory_path, min_count, num_workers):
    func = partial(build_vocabs, min_count=min_count)
    p = multiprocessing.Pool(num_workers)
    word_counter = Counter()
    char_counter = Counter()
    char_counter.update(['{', '}'])
    counter = 0
    for x, y in p.imap(func, directory_path):
        counter += 1
        print(counter)
        word_counter += x
        char_counter += y
    return word_counter, char_counter


def build_vocabs(filepath, min_count):
    """Build the word and char counter vocabularies"""
    toktok = ToktokTokenizer()
    word_vocab = Counter()
    char_vocab = Counter()
    with open(filepath, 'r', encoding='utf8') as f:
        try:
            line = f.read()
            if 'numbers_' in filepath:
                tmp = toktok.tokenize(line.lower())
                for i in range(min_count):
                    word_vocab.update(tmp)
            else:
                word_vocab.update(word_tokenize(line.lower()))
            char_vocab.update(line)
        except Exception as error:
            print('Error with file: {}'.format(filepath))
            print(error)
    return word_vocab, char_vocab


def build_utilities(word_vocab, char_vocab, vocab_size, min_cnt):
    mst_commn = [e[0] for e in char_vocab.most_common(vocab_size-4)]
    char_list = ['PAD', '{', '}'] + mst_commn + ['UNK']
    char_to_index = {e: n for n, e in enumerate(char_list)}
    index_to_char = {n: e for n, e in enumerate(char_list)}

    word_vocab = {w: n for w, n in word_vocab.items()
                  if (n >= min_cnt) & (len(w) <= 30)}
    num_total_words = sum([num for word, num in word_vocab.items()])
    unigram_table = []
    Z = 0.001
    for word in word_vocab:
        tmp = word_vocab[word]/num_total_words
        unigram_table.extend([word] * int(((tmp)**0.75)/Z))

    return dict(char_to_index=char_to_index, index_to_char=index_to_char,
                unigram_table=unigram_table, num_total_words=num_total_words,
                word_vocab=word_vocab)


def file_to_features(path, word_vocab, window, min_count, total_w):
    examples = []
    toktok = ToktokTokenizer()
    punckt = set(string.punctuation)
    try:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                for sentence in sent_tokenize(line):
                    words_1 = toktok.tokenize(sentence)
                    words_2 = []
                    for i, word in enumerate(words_1):
                        word_l = word.lower()
                        if word_l not in word_vocab:
                            continue
                        if word_vocab[word_l] < min_count:
                            continue
                        if word in punckt:
                            continue
                        frequency = word_vocab[word_l] / total_w
                        number = 1 - math.sqrt(10e-5/frequency)
                        if random.uniform(0, 1) <= number:
                            continue
                        words_2.append(word)
                    max_j = len(words_2)
                    for i, word in enumerate(words_2):
                        start = i - window if (i - window) > 0 else 0
                        to = i + window if (i + window) < max_j else max_j
                        for j in range(start, to):
                            if i == j:
                                continue
                            target = words_2[j]
                            examples.append((word, target))
    except Exception as error:
        print(error)
    return examples


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build_dataset(paths, num_workers, word_vocab, min_count,
                  window, num_total_words, archive):
    func = partial(file_to_features, word_vocab=word_vocab, window=window,
                   min_count=min_count, total_w=num_total_words)
    p = multiprocessing.Pool(num_workers, init_worker)
    files = []
    file_counter = 0
    filename = archive.format(file_counter)
    files.append(filename)
    archive_f = open(archive.format(file_counter), 'w', encoding='utf-8')
    archive_f.write('Source\tTarget\n')
    counter = 0
    for x in p.imap(func, paths):
        if counter % 100 == 0:
            print(counter)
        counter += 1
        for word, target in x:
            archive_f.write('{}\t{}\n'.format(word, target))
        if archive_f.tell() > 5e+8:
            print('Changing file. Counter at {}'.format(counter))
            file_counter += 1
            filename = archive.format(file_counter)
            files.append(filename)
            archive_f = open(archive.format(file_counter), 'w',
                             encoding='utf-8')
    archive_f.close()
    return files
