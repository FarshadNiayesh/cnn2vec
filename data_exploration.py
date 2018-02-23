from model import Word2CNN
from data_preperation import prepare_tensor, build_utilities
from torch.autograd import Variable
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import argparse

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main(args):
    checkpoint = torch.load(args.model_path)
    model_params = checkpoint['args']
    model = Word2CNN(model_params.vocab_size, model_params.char_embed,
                     model_params.kernel_numbers, model_params.kernel_sizes)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda() if USE_CUDA else model

    if (args.word_path is not None) and (args.char_path is not None):
        word_vocab = load_obj(args.word_path)
        char_vocab = load_obj(args.char_path)
    else:
        raise RuntimeError

    tmp = build_utilities(word_vocab, char_vocab, model_params.vocab_size)
    c_to_index = tmp[0]
    num_total_words = tmp[3]
    # index_to_char = tmp[1]
    # unigram_table = tmp[2]
    similarities = np.zeros((num_total_words, len(args.words)))
    vocab_list = [word for word, num in word_vocab.items()]
    embed_words = []
    for i, word in enumerate(args.words):
        word = Variable(LongTensor(prepare_tensor(word, c_to_index)))
        word = word.unsqueeze(0)
        embed_words.append(model.prediction(word))

    for i, word in enumerate(vocab_list):
        if i % 1000 == 0:
            print('{} words remaining.'.format(i), end='\r')
        target = prepare_tensor(word, c_to_index)
        target = Variable(LongTensor(target)).unsqueeze(0)
        target_e = model.prediction(target)
        for j, embed in enumerate(embed_words):
            sim = F.cosine_similarity(embed, target_e)
            similarities[i, j] = sim
    print('')

    for i, word in enumerate(args.words):
        sim = similarities[:, i]
        indices = np.argsort(sim)
        print('Similarities for {}'.format(word))
        for i in indices[:args.num_sim]:
            print(vocab_list[i])
        print('-----')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--words', help='Words to check for similarities',
                        nargs="+", type=str, default=['Financeeeeee'])
    parser.add_argument('--num_sim', help='Number of similarities to display',
                        default=5)
    parser.add_argument('--model_path', help='Where to find the last checkpnt',
                        default="assets/checkpoint.pth.tar")
    parser.add_argument('--word_path', help='Where to find word vocab',
                        default='assets/word_vocab.pkl')
    parser.add_argument('--char_path', help='Where to find char vocab',
                        default='assets/char_vocab.pkl')
    args = parser.parse_args()
    main(args)
