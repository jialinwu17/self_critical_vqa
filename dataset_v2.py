import os
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

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

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
            ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
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
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


class SelfCriticalDataset(Dataset):
    def __init__(self, name, dictionary, opt, dataroot='data'):
        super(SelfCriticalDataset, self).__init__()
        assert name in ['v2cp_train', 'v2cp_test', 'vg', 'vgcp_train', 'vgcp_test', 'vgcp_train_hat', 'vgcp_test_hat',
                        'v2cp_train_hat', 'v2cp_test_hat', 'v2cp_train_hatvqx', 'v2cp_test_hatvqx', 'v2cp_train_vqx',
                        'v2cp_test_vqx']
        self.name = name
        self.dictionary = dictionary  # questions' dictionary

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        if 'vg' not in self.name:
            self.trainimg_id2idx = cPickle.load(open(os.path.join(dataroot, '%s36_imgid2img.pkl' % 'train'), 'rb'))
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % 'train')
            self.train_hf = h5py.File(h5_path, 'r')
            self.train_features = self.train_hf.get('image_features')
            self.train_spatials = self.train_hf.get('spatial_features')
            self.train_cls_scores = np.array(self.train_hf.get('cls_score'))
            self.train_attr_scores = np.array(self.train_hf.get('attr_score'))

            self.valimg_id2idx = cPickle.load(open(os.path.join(dataroot, '%s36_imgid2img.pkl' % 'val'), 'rb'))
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % 'val')
            self.val_hf = h5py.File(h5_path, 'r')
            self.val_features = self.val_hf.get('image_features')
            self.val_spatials = self.val_hf.get('spatial_features')
            self.val_cls_scores = np.array(self.val_hf.get('cls_score'))
            self.val_attr_scores = np.array(self.val_hf.get('attr_score'))
        else:
            self.trainimg_id2idx = cPickle.load(open(os.path.join(dataroot, '%s36_imgid2img.pkl' % 'vg'), 'rb'))
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % 'vg')
            self.train_hf = h5py.File(h5_path, 'r')
            self.train_features = self.train_hf.get('image_features')
            self.train_spatials = self.train_hf.get('spatial_features')
            self.train_cls_scores = np.array(self.train_hf.get('cls_score'))
            self.train_attr_scores = np.array(self.train_hf.get('attr_score'))

        if name == 'vg' or name == 'v2cp_test' or name == 'v2cp_train':
            if name == 'vg':
                self.entriess = cPickle.load(open(dataroot + '/VG_dataset.pkl', 'rb'))
            else:
                if name == 'v2cp_test':
                     self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'val' + 'dataset.pkl', 'rb'))
                else:
                    self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'train' + 'dataset.pkl', 'rb'))
            self.entries = {}
            count = 0
            self.train_hint = cPickle.load(open(os.path.join(dataroot, 'train_v2_hint.pkl'), 'rb'))
            self.val_hint = cPickle.load(open(os.path.join(dataroot, 'val_v2_hint.pkl'), 'rb'))

            for i in tqdm(range(len(self.entriess))):
                if len(self.entriess[i]['question']) == 0:
                    continue
                imgid = int(self.entriess[i]['question'][0]['image_id'])
                for k in range(len(self.entriess[i]['question'])):

                    question = self.entriess[i]['question'][k]['question'].lower()
                    if ('how many' in question) or ('how much' in question):
                        if opt.use_all == 0 :
                            continue
                    if self.entriess[i]['question'][k]['question_id'] in self.val_hint.keys():
                        hint = self.val_hint[self.entriess[i]['question'][k]['question_id']]
                        hint_a = np.zeros((36))
                        obj_cls = np.array(self.val_cls_scores[self.valimg_id2idx[imgid]][:, 0])
                        hint_o = obj_cls.astype(
                            'float')  # hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                    elif self.entriess[i]['question'][k]['question_id'] in self.train_hint.keys():
                        hint = self.train_hint[self.entriess[i]['question'][k]['question_id']]
                        hint_a = np.zeros((36))
                        obj_cls = np.array(self.train_cls_scores[self.trainimg_id2idx[imgid]][:, 0])
                        hint_o = obj_cls.astype('float')  # (hint > 0.2).astype('float') * obj_cls.astype('float')
                    else:
                        if opt.use_all == 0:
                            continue
                        else:
                            hint_a = np.zeros((36))
                            hint_o = np.zeros((36))
                            hint = np.zeros((36))

                    if imgid in self.trainimg_id2idx:
                        new_entry = {'image': self.trainimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    else:
                        new_entry = {'image': self.valimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    self.entries[count] = new_entry
                    count += 1
        elif name == 'vgcp_train' or name == 'vgcp_test':
            self.entriess = pickle.load(open('data/%s_entries.pkl'%name, 'rb'))
            count = 0
            self.entries = {}
            print('Preparing Entries!!!')
            for i in range(len(self.entriess)):
                new_entry = self.entriess[i]
                new_entry['hint'] = np.zeros((36))
                new_entry['hint_a'] = np.zeros((36))
                new_entry['hint_o'] = np.zeros((36))
                self.entries[count] = new_entry
                assert new_entry['image'] == self.trainimg_id2idx[new_entry['image_id']]
                count += 1

        elif name == 'vgcp_train_hat' or name == 'vgcp_test_hat':
            self.entriess = pickle.load(open('data/%s_entries.pkl'%('_'.join(name.split('_')[:-1])), 'rb'))
            count = 0
            self.entries = {}
            for i in range(len(self.entriess)):
                qid = self.entriess[i]['question_id']

                if not os.path.isfile('data/VG_qid2knowledge/%s.pkl' % str(qid)):
                    continue
                else:
                    hints = cPickle.load(open('data/VG_qid2knowledge/%s.pkl' % str(qid), 'rb'))
                    hint_objects = hints['hint_objects']
                    hint_attributes = hints['hint_attributes']
                    hint = (hint_objects > 0).astype('float')
                    hint_a = hint_attributes.astype('float')
                    hint_o = hint_objects.astype('float')
                    new_entry = self.entriess[i]
                    new_entry['hint'] = hint
                    new_entry['hint_a'] = hint_a
                    new_entry['hint_o'] = hint_o
                    self.entries[count] = new_entry
                    count += 1

        elif name == 'v2cp_test_hat' or name == 'v2cp_train_hat':
            count = 0
            self.entries = {}
            if name == 'v2cp_test_hat':
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'val' + 'dataset.pkl', 'rb'))
            else:
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'train' + 'dataset.pkl', 'rb'))
            self.train_hint = cPickle.load(open(os.path.join(dataroot, 'train_qid2hint.pkl'), 'rb'))
            self.val_hint = cPickle.load(open(os.path.join(dataroot, 'val_qid2hint.pkl'), 'rb'))
            for i in tqdm(range(len(self.entriess))):
                if len(self.entriess[i]['question']) == 0:
                    continue
                imgid = int(self.entriess[i]['question'][0]['image_id'])
                for k in range(len(self.entriess[i]['question'])):
                    if self.entriess[i]['question'][k]['question_id']:
                        question = self.entriess[i]['question'][k]['question'].lower()
                        if ('how many' in question) or ('how much' in question):
                            continue
                        if self.entriess[i]['question'][k]['question_id'] in self.val_hint.keys():
                            hint = self.val_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.val_cls_scores[self.valimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                        elif self.entriess[i]['question'][k]['question_id'] in self.train_hint.keys():
                            hint = self.train_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.train_cls_scores[self.trainimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') # (hint > 0.2).astype('float') * obj_cls.astype('float')
                        else:
                            continue

                    if imgid in self.trainimg_id2idx:
                        new_entry = {'image': self.trainimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    else:
                        new_entry = {'image': self.valimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    self.entries[count] = new_entry
                    count += 1
        elif name == 'v2cp_test_vqx' or name == 'v2cp_train_vqx':
            count = 0
            self.entries = {}
            if name == 'v2cp_test_vqx':
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'val' + 'dataset.pkl', 'rb'))
            else:
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'train' + 'dataset.pkl', 'rb'))
            self.train_hint = cPickle.load(open(os.path.join(dataroot, 'train_vqx_hint.pkl'), 'rb'))
            self.val_hint = cPickle.load(open(os.path.join(dataroot, 'val_vqx_hint.pkl'), 'rb'))

            for i in tqdm(range(len(self.entriess))):
                if len(self.entriess[i]['question']) == 0:
                    continue
                imgid = int(self.entriess[i]['question'][0]['image_id'])
                for k in range(len(self.entriess[i]['question'])):
                    if self.entriess[i]['question'][k]['question_id']:
                        if self.entriess[i]['question'][k]['question_id'] in self.val_hint.keys():
                            hint = self.val_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.val_cls_scores[self.valimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        elif self.entriess[i]['question'][k]['question_id'] in self.train_hint.keys():
                            hint = self.train_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.train_cls_scores[self.trainimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        else:
                            continue
                    if imgid in self.trainimg_id2idx:
                        new_entry = {'image': self.trainimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    else:
                        new_entry = {'image': self.valimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    self.entries[count] = new_entry
                    count += 1
        elif name == 'v2cp_test_hatvqx' or name == 'v2cp_train_hatvqx':
            count = 0
            self.entries = {}
            if name == 'v2cp_test_hatvqx':
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'val' + 'dataset.pkl', 'rb'))
            else:
                self.entriess = cPickle.load(open(dataroot + '/VQA_caption_' + 'train' + 'dataset.pkl', 'rb'))
            self.train_hint = cPickle.load(open(os.path.join(dataroot, 'train_vqx_hint.pkl'), 'rb'))
            self.val_hint = cPickle.load(open(os.path.join(dataroot, 'val_vqx_hint.pkl'), 'rb'))
            self.train_hint_hat = cPickle.load(open(os.path.join(dataroot, 'train_qid2hint.pkl'), 'rb'))
            self.val_hint_hat = cPickle.load(open(os.path.join(dataroot, 'val_qid2hint.pkl'), 'rb'))

            for i in tqdm(range(len(self.entriess))):
                if len(self.entriess[i]['question']) == 0:
                    continue
                imgid = int(self.entriess[i]['question'][0]['image_id'])
                for k in range(len(self.entriess[i]['question'])):
                    if self.entriess[i]['question'][k]['question_id']:
                        if self.entriess[i]['question'][k]['question_id'] in self.val_hint_hat.keys():
                            hint = self.val_hint_hat[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.val_cls_scores[self.valimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        elif self.entriess[i]['question'][k]['question_id'] in self.train_hint_hat.keys():
                            hint = self.train_hint_hat[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.train_cls_scores[self.trainimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        elif self.entriess[i]['question'][k]['question_id'] in self.val_hint.keys():
                            hint = self.val_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.val_cls_scores[self.valimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        elif self.entriess[i]['question'][k]['question_id'] in self.train_hint.keys():
                            hint = self.train_hint[self.entriess[i]['question'][k]['question_id']]
                            hint_a = np.zeros((36))
                            obj_cls = np.array(self.train_cls_scores[self.trainimg_id2idx[imgid]][:, 0])
                            hint_o = obj_cls.astype('float') #hint_o = (hint > 0.2).astype('float') * obj_cls.astype('float')
                            #hint_o = np.zeros((36))
                        else:
                            continue
                    if imgid in self.trainimg_id2idx:
                        new_entry = {'image': self.trainimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    else:
                        new_entry = {'image': self.valimg_id2idx[imgid],
                                     'image_id': imgid,
                                     'question_id': self.entriess[i]['question'][k]['question_id'],
                                     'question': self.entriess[i]['question'][k]['question'],
                                     'answer': self.entriess[i]['answer'][k],
                                     'hint': hint, 'hint_a': hint_a, 'hint_o': hint_o}
                    self.entries[count] = new_entry
                    count += 1

        self.tokenize()
        self.tensorize()
        self.v_dim = 2048  # self.features.size(2)
        self.s_dim = 36  # self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if labels is None:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
            elif len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        # features
        entry = self.entries[index]
        imgid = self.entries[index]['image_id']
        qid = self.entries[index]['question_id']
        if imgid in self.trainimg_id2idx:
            obj_nodes = torch.from_numpy(np.array(self.train_features[entry['image']]))
        else:
            obj_nodes = torch.from_numpy(np.array(self.val_features[entry['image']]))

        hint_score = torch.from_numpy(entry['hint'])
        hint_o = torch.from_numpy(entry['hint_o'])

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        hint_score = hint_score.float().unsqueeze(1)

        if labels is not None:
            target.scatter_(0, labels, scores)

        return obj_nodes, question, target, hint_score, hint_o, qid

    def __len__(self):
        return len(self.entries)
