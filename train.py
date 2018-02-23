from model import Word2CNN
from data_preperation import build_utilities, SingleFileDataset
from data_preperation import DataLoaderMultiFiles
import torch
from torch.autograd import Variable
from functools import partial
import torch.optim as optim
import numpy as np
import shutil
import pickle
import argparse
import os
import random
random.seed(0)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='assets/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'assets/model_best.pth.tar')


def main(args):
    if (args.word_path is not None) and (args.char_path is not None):
        word_vocab = load_obj(args.word_path)
        char_vocab = load_obj(args.char_path)
    else:
        pass  # TODO

    tmp = build_utilities(word_vocab, char_vocab, args.vocab_size)
    char_to_index = tmp[0]
    unigram_table = tmp[2]
    num_total_words = tmp[3]

    cwd = os.getcwd()
    corpus = args.corpus_dir
    files = [os.path.join(cwd, corpus, e) for e in os.listdir(corpus)]
    files = files[:args.max_files]
    partialfn = partial(SingleFileDataset, word_vocab=word_vocab,
                        char_to_index=char_to_index, min_count=args.min_count,
                        window=args.window, unigram_table=unigram_table,
                        num_total_words=num_total_words,
                        neg_samples=args.num_negs)
    vocab_size = len(char_to_index)
    model = Word2CNN(vocab_size, args.char_embed, args.kernel_numbers,
                     args.kernel_sizes)
    model = model.cuda() if USE_CUDA else model
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dl = DataLoaderMultiFiles(files, partialfn, args.batch, args.buffer_size)
    min_loss = np.inf
    for epoch in range(args.epochs):
        losses = []
        epoch_losses = []
        for i, batch in enumerate(dl):
            inputs = Variable(batch, requires_grad=False)
            loss = model(inputs, args.num_negs)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            losses.append(loss.data.tolist()[0])
            if (i % 100 == 0) and (i > 0):
                mean_l = np.mean(losses)
                epoch_losses.append(mean_l)
                print("[%d/%d] mean_loss : %0.2f" % (epoch, args.epochs,
                      mean_l))
                losses = []
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'min_loss': min_loss,
                                 'optimizer': optimizer.state_dict(),
                                 'learning_rate': lr,
                                 'args': args}, False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--word_path', help='Word counter relative filepath',
                        default="assets/word_vocab.pkl")
    parser.add_argument('--char_path', help='Characters counter filepath',
                        default="assets/char_vocab.pkl")
    parser.add_argument('--ds_path', help='DataSet Path', default=None)
    parser.add_argument('--batch', help='Batch size', type=int, default=2048)
    parser.add_argument('--window', help='Skip-grame window', default=3,
                        type=int)
    parser.add_argument('--num_negs', help='Number of negatives samples',
                        default=10, type=int)
    parser.add_argument('--epochs', help='Number of epochs', type=int,
                        default=5)
    parser.add_argument('--char_embed', help='Embedding size of characters',
                        default=15, type=int)
    parser.add_argument('--lr', help='Learning rate', default=0.0001,
                        type=float)
    parser.add_argument('--min_count', help='Min number of occur. for words',
                        default=2, type=int)
    parser.add_argument('--kernel_sizes', help='Width of conv. kernels',
                        default=[1, 2, 3, 4, 5, 6, 7], nargs='+', type=int)
    parser.add_argument('--kernel_numbers', help='Number of kernels per conv.',
                        default=[50, 100, 150, 200, 200, 200, 200], nargs='+',
                        type=int)
    parser.add_argument('--corpus_dir', help='Directory with text files',
                        default="corpus")
    parser.add_argument('--max_files', help='Maximum number of files to load',
                        type=int, default=10000000000)
    parser.add_argument('--num_workers', help='Number of DataLoader workers',
                        type=int, default=0)
    parser.add_argument('--buffer_size', help='Data buffer size',
                        type=int, default=10000)
    parser.add_argument('--vocab_size', help='Char vocab size',
                        type=int, default=125)
    args = parser.parse_args()
    main(args)
