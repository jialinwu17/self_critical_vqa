from __future__ import print_function
import os
import sys, pickle
import json
import pickle as cPickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize_elmo(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        return words

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                if '-' in w:
                    print(w)
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def create_dictionary():
    dictionary = Dictionary()
    questions = []
    qid2exp = {}
    files = [
        'VQA-X/textual/test_exp_anno.json',
        'VQA-X/textual/train_exp_anno.json',
        'VQA-X/textual/val_exp_anno.json']
    for path in files:
        question_path = os.path.join('', path)
        qs = json.load(open(question_path))
        keys = qs.keys()
        for k in keys:
            dictionary.tokenize(qs[k][0], True)
            qid2exp[int(k)] = qs[k][0]
    return dictionary, qid2exp


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        #vals = map(float, vals[1:])
        valv = [float(v) for v in vals[1:]]
        word2emb[word] = np.array(valv)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d, qid2exp = create_dictionary()
    d.dump_to_file('vqxdictionary.pkl')

    d = Dictionary.load_from_file('vqxdictionary.pkl')
    emb_dim = 300
    glove_file = 'glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('glove6b_init_vqx_%dd.npy' % emb_dim, weights)
    pickle.dump(qid2exp, open('qid2exp.pkl', 'wb'))

