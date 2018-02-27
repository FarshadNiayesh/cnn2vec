from model import Word2CNN
from data_preperation import build_utilities, SingleFileDataset, build_vocabs
from data_preperation import DataLoaderMultiFiles, build_dataset, build_vocab_multi
import torch
from torch.autograd import Variable
from functools import partial
import torch.optim as optim
import numpy as np
import pickle
import argparse
import os
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tensorboardX import SummaryWriter
import klepto

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
# writer = SummaryWriter()


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state):
    if state['end_of_epoch']:
        e = state['epoch']
        filename = 'assets/checkpoints/end_epoch_{}.pth.tar'.format(e)
    else:
        e = state['epoch']
        filename = 'assets/checkpoints/run_epoch_{}.pth.tar'.format(e)
    torch.save(state, filename)


def main(args):

    # Utilities building
    if (args.word_path is not None) and (args.char_path is not None):
        word_vocab = load_obj(args.word_path)
        char_vocab = load_obj(args.char_path)
    else:
        cwd = os.getcwd()
        corpus = args.corpus_dir
        files = [os.path.join(cwd, corpus, e) for e in os.listdir(corpus)]
        files = files[:args.max_files]
        word_vocab, char_vocab = build_vocab_multi(files, args.min_count, 4)
        save_obj(word_vocab, 'assets/word_vocab.pkl')
        save_obj(char_vocab, 'assets/char_vocab.pkl')

    tmp = build_utilities(word_vocab, char_vocab, args.vocab_size,
                          args.min_count)
    char_to_index = tmp['char_to_index']
    unigram_table = tmp['unigram_table']
    num_total_words = tmp['num_total_words']
    word_vocab = tmp['word_vocab']
    vocab_size = len(char_to_index)

    if args.dataset is not None:
        dataset = load_obj(args.dataset)
    else:
        cwd = os.getcwd()
        corpus = args.corpus_dir
        files = [os.path.join(cwd, corpus, e) for e in os.listdir(corpus)]
        files = files[:args.max_files]
        archive = klepto.archives.dir_archive('archive', cached=False)
        dataset = build_dataset(files, 4, word_vocab, char_vocab,
                                args.min_count, args.window, unigram_table,
                                num_total_words, args.num_negs, archive)
        save_obj(dataset, 'assets/dataset.pkl')

    # Model intialization
    model = Word2CNN(vocab_size, args.char_embed, args.kernel_numbers,
                     args.kernel_sizes, batch_norm=args.batch_norm)
    model = model.cuda() if USE_CUDA else model

    lr = args.lr
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
        scheduler = MultiStepLR(optimizer, milestones=[1e3, 1e4, 1e5, 1e6],
                                gamma=0.5)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    elif args.optim == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    if args.resume:
        checkpoint = torch.load(args.archive_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch']
        lr = checkpoint['learning_rate']
        min_loss = checkpoint['min_loss']
    else:
        current_epoch = 0
        lr = args.lr
        min_loss = np.inf

    dl = DataLoaderMultiFiles(dataset, args.batch)
    for epoch in range(current_epoch, args.epochs):
        losses = []
        mean_losses = [np.inf]
        for i, batch in enumerate(dl):
            if args.optim == "sgd":
                scheduler.step()
            inputs = Variable(batch, requires_grad=False)
            loss = model(inputs, args.num_negs)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss = loss.data.tolist()[0]
            losses.append(loss)
            if i % args.report == 0:
                mean_l = np.mean(losses[-args.report:])
                # writer.add_scalar('data/interim_loss', mean_l, (epoch+1)*i)
                if mean_l < np.min(mean_losses):
                    save_checkpoint({'epoch': epoch,
                                     'state_dict': model.state_dict(),
                                     'min_loss': min_loss,
                                     'loss': mean_l,
                                     'end_of_epoch': False,
                                     'optimizer': optimizer.state_dict(),
                                     'learning_rate': lr,
                                     'args': args})
                mean_losses.append(mean_l)
        epoch_loss = np.mean(losses)
        print('Epoch loss: {}'.format(epoch_loss))
        # writer.add_scalar('data/epoch_loss', epoch_loss, epoch)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'min_loss': min_loss,
                         'loss': mean_l,
                         'end_of_epoch': True,
                         'optimizer': optimizer.state_dict(),
                         'learning_rate': lr,
                         'args': args})



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--word_path', help='Word counter relative filepath',
                        default=None) #  assets/word_vocab.pkl
    parser.add_argument('--char_path', help='Characters counter filepath',
                        default=None) #  "assets/char_vocab.pkl"
    parser.add_argument('--ds_path', help='DataSet Path', default=None)
    parser.add_argument('--batch', help='Batch size', type=int, default=1024)
    parser.add_argument('--window', help='Skip-gram window', default=5,
                        type=int)
    parser.add_argument('--num_negs', help='Number of negatives samples',
                        default=5, type=int)
    parser.add_argument('--epochs', help='Number of epochs', type=int,
                        default=5)
    parser.add_argument('--char_embed', help='Embedding size of characters',
                        default=15, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-3,
                        type=float)
    parser.add_argument('--min_count', help='Min number of occur. for words',
                        default=5, type=int)
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
                        type=int, default=750000)
    parser.add_argument('--vocab_size', help='Char vocab size',
                        type=int, default=125)
    parser.add_argument('--optim', help='Optimizer (sgd, adam)', default='sgd')
    parser.add_argument('--report', help='Report stats every X batch.',
                        default=100, type=int)
    parser.add_argument('--resume', help="Resume training",
                        action='store_true')
    parser.add_argument('--archive_path', help="Path where to find the model")
    parser.add_argument('--batch_norm', help="Use batchnorm (default False)",
                        action='store_true')
    parser.add_argument('--dataset', help="Path to the pickled datasets",
                        default=None)
    args = parser.parse_args()
    main(args)
