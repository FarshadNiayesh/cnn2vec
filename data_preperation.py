import torch
import numpy as np
from nltk import word_tokenize
from nltk.tokenize import ToktokTokenizer
import math
import random
import os
from collections import Counter
import torch.multiprocessing
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from queue import Empty
from datetime import datetime

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(0)
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class SingleFileDataset(object):
    """Dataset to prepare a single file of Skip-Gram entries"""

    def __init__(self, path, word_vocab, char_to_index, min_count, window,
                 unigram_table, num_total_words, neg_samples=5):
        self.min_count = min_count
        self.path = path
        self.w_vocab = word_vocab
        self.c_to_index = char_to_index
        self.unk = char_to_index['UNK']
        self.total_w = num_total_words
        self.window = window
        self.table = unigram_table
        self.neg_samples = neg_samples

    def next(self):
        """Get an item from the examples and add negative samples"""
        for word, target in self.prepare_file(self.path):
            negs = self.get_negatives(word)
            negs = [self.prepare_tensor(neg) for neg in negs]
            example = [self.prepare_tensor(word), self.prepare_tensor(target)]
            example += negs
            yield example

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

    def prepare_file(self, filepath):
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                for word, target in self.prepare_line(line):
                    yield word, target

    def prepare_line(self, line):
        words = word_tokenize(line)
        max_j = len(words)
        for i, word in enumerate(words):
            if word.lower() not in self.w_vocab:
                continue
            if self.w_vocab[word.lower()] < self.min_count:
                continue
            frequency = self.w_vocab[word.lower()] / self.total_w
            number = 1 - math.sqrt(10e-5/frequency)
            if random.uniform(0, 1) <= number:
                continue
            for j in range(i - self.window, i + self.window):
                if (i == j) or (j < 0) or (j >= max_j):
                    continue
                target = words[j]
                if target.lower() not in self.w_vocab:
                    continue
                yield word, target


class DataLoaderMultiFiles(object):
    """DataLoader to iterator over a set of DataSet"""

    def __init__(self, filepaths, partial, batch_s, buffer_s):
        self.filepaths = filepaths
        self.partial = partial
        self.batch_size = batch_s
        self.max_len = buffer_s
        self.buffer = Queue(maxsize=buffer_s)
        self.batch_queue = Queue(maxsize=10)

    def __iter__(self):
        print('Starting processes')
        args = (self.filepaths, self.buffer, self.partial)
        self.buffr_process = Process(target=fill_buffer, args=args)
        self.buffr_process.daemon = True
        self.buffr_process.start()
        args = (self.buffer, self.batch_queue, self.batch_size)
        self.batch_process = Process(target=fill_batch, args=args)
        self.batch_process.daemon = True
        self.batch_process.start()
        return self

    def __next__(self):
        # print('get batch')
        # print('buffer_queue: {}, batch_queue: {}'.format(self.buffer.qsize(), self.batch_queue.qsize())) # noqa
        try:
            batch = self.batch_queue.get(timeout=60)
            if type(batch) == str:
                if batch == 'DONE':
                    raise StopIteration
        except Empty:
            self.kill()
            raise StopIteration
        # print('got batch')
        tmp = LongTensor(batch)
        # print('computing')
        return tmp

    def kill(self):
        print('Killing processes')
        self.buffr_process.terminate()
        self.batch_process.terminate()


def fill_buffer(filepaths, buffr, partial):
    pid = torch.multiprocessing.current_process()._identity[0]
    print('Seed of {}'.format(pid))
    random.seed(pid)
    random.shuffle(filepaths)
    start_time = datetime.now()
    ln = len(filepaths)
    for i, path in enumerate(filepaths):
        if (i % 100 == 0) and (i > 0):
            diff = (datetime.now() - start_time).seconds
            if diff == 0:
                continue
            time_per_doc = i/diff
            time_left = (ln - i)*time_per_doc/3600
            print('{:.2f} hours until new epoch'.format(time_left))
        ds = partial(path)
        for element in ds.next():
            buffr.put(element)
    print('End of datasets')
    buffr.put('DONE')


def max_size(elements):
    max_size = 0
    for elem in elements:
        if len(elem) > max_size:
            max_size = len(elem)
    return max_size


def fill_batch(buffr, batch, batch_size):
    pid = torch.multiprocessing.current_process()._identity[0]
    print('Seed of {}'.format(pid))
    go_on = True
    while go_on:
        # print('New batch, queue size: {}'.format(buffr.qsize()))
        # deque the batch
        if buffr.qsize() == 0:
            print('Empty queue!')
            go_on = False
            continue
        internal_buffer = []
        for i in range(buffr.qsize()):
            tmp = buffr.get()
            if tmp == 'DONE':
                go_on = False
                break
            internal_buffer.append(tmp)
        # internal_buffer = sorted(internal_buffer, key=max_size)
        random.shuffle(internal_buffer)
        for i in range(0, len(internal_buffer), batch_size):
            elements = internal_buffer[i:i+batch_size]
            current_batch = []
            for ele in elements:
                current_batch.extend(ele)
            batch.put(pad_sequences(current_batch))
    batch.put('DONE')


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


def build_vocabs(directory_path, min_count):
    """Build the word and char counter vocabularies"""
    toktok = ToktokTokenizer()
    word_vocab = Counter()
    char_vocab = Counter()
    char_vocab.update(['{', '}'])
    filenames = os.listdir(directory_path)
    filepaths = [os.path.join(directory_path, e) for e in filenames]
    for i, filepath in enumerate(filepaths):
        if i % 100 == 0:
            print('Reading file number {}'.format(i), end="\r")
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
