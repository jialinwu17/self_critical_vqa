import cv2, pickle, h5py, os, spacy, json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle as cPickle

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
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', 
            ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        return words

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


ans2label = pickle.load(open('cache/trainval_ans2label.pkl', 'rb'))
label2ans = pickle.load(open('cache/trainval_label2ans.pkl', 'rb'))

qid2q = {}
qimgid2qid = {}
aaa = pickle.load(open('VQA_caption_valdataset.pkl', 'rb'))
imgid2caption = {}
qid2ans = {}

for i in tqdm(range(len(aaa))):
    ents = aaa[i]['question']
    imgid2caption[ents[0]['image_id']] = aaa[i]['caption']

    for j in range(len(ents)):
        ent = ents[j]
        question = ent['question']
        qid = ent['question_id']
        qid2q[qid] = question
        qid2ans[qid] = aaa[i]['answer'][j]
        qimgid2qid[question + ',' + str(ents[0]['image_id'])] = qid

bbb = pickle.load(open('VQA_caption_traindataset.pkl', 'rb'))
for i in tqdm(range(len(bbb))):
    ents = bbb[i]['question']
    imgid2caption[ents[0]['image_id']] = bbb[i]['caption']
    for j in range(len(ents)):
        ent = ents[j]
        question = ent['question']
        qid = ent['question_id']
        qid2q[qid] = question
        qid2ans[qid] = bbb[i]['answer'][j]
        qimgid2qid[question + ',' + str(ents[0]['image_id'])] = qid

names = ['train', 'val']
nlp = spacy.load('en_core_web_lg')

for name in names:
    qid2hint = {}
    img_id2idx = pickle.load(open(os.path.join('%s36_imgid2img.pkl' % name), 'rb'))
    h5_path = os.path.join('%s36.hdf5' % name)
    hf = h5py.File(h5_path, 'r')
    features = hf.get('image_features')
    spatials = hf.get('spatial_features')
    cls_scores = np.array(hf.get('cls_score'))
    attr_scores = np.array(hf.get('attr_score'))
    
    qid2exp = pickle.load(open('qid2exp.pkl', 'rb'))
    qids = list(qid2exp.keys())
    obj_emb = np.load('glove6b_init_objects_300d.npy')
    exp_emb = np.load('glove6b_init_vqx_300d.npy')
    attr_emb = np.load('glove6b_init_attributes_300d.npy')

    d = Dictionary.load_from_file('vqxdictionary.pkl')
    for i in tqdm(range(len(qids))):
        qid = qids[i]
        
        imgid = qid // 1000
        newqid = qid 
        if imgid not in img_id2idx:
            continue

        img = img_id2idx[imgid]
        boxes = spatials[img, :, :4]
        H = spatials[img, 0, 5]
        W = spatials[img, 0, 6]

        clss = cls_scores[img_id2idx[imgid]][:, 0].astype('int')
        attrs = attr_scores[img_id2idx[imgid]][:, 0].astype('int')

        exp = qid2exp[qid]
        tokens = d.tokenize(exp, 0)

        doc = nlp(exp)
        object_attributes = []
        for token in doc:
            if token.pos_ == u'NUM' and token.dep_ == u'nummod' and token.head.pos_ == u'NOUN':
                obj = token.head.text
                attri = token.text
                key = str(obj) + ',' + str(attri)
                object_attributes.append(key)
            elif token.pos_ == u'ADJ' and token.dep_ == u'amod' and token.head.pos_ == u'NOUN':
                obj = token.head.text
                attri = token.text
                key = str(obj) + ',' + str(attri)
                object_attributes.append(key)
            elif token.pos_ == u'NOUN' and token.dep_ == u'compound' and token.head.pos_ == u'NOUN':
                obj = token.head.text
                attri = token.text
                key = str(obj) + ',' + str(attri)
                object_attributes.append(key)


        objs = obj_emb[clss, :]
        atts = attr_emb[attrs, :]
        exps = exp_emb[tokens]
        obj_sim = cosine_similarity(objs, exps)
        attr_sim = cosine_similarity(atts, exps)

        hint_score = np.zeros((36))
        hint_score_attr = np.zeros((36))

        mentioned_tokens = []
        mentioned_attrs = []

        for j in range(36):
            for k in range(len(tokens)):
                if obj_sim[j, k] > 0.6:
                    if hint_score[j] <= obj_sim[j, k]:
                        hint_score[j] = obj_sim[j, k]
                    if d.idx2word[tokens[k]] not in mentioned_tokens:
                        mentioned_tokens.append(d.idx2word[tokens[k]])
            for key in object_attributes:
                obj, attr = key.split(',')
                obj_token = d.tokenize(obj, 0)[0]
                attr_token = d.tokenize(attr, 0)[0]
                if cosine_similarity(exp_emb[obj_token:obj_token+1], objs[j:j+1]) > 0.3 :
                    if cosine_similarity(exp_emb[attr_token:attr_token+1], atts[j:j+1]) > 0.3 :
                        if hint_score_attr[j] <= cosine_similarity(exp_emb[attr_token:attr_token+1], atts[j:j+1]):
                            hint_score[j] = cosine_similarity(exp_emb[attr_token:attr_token+1], atts[j:j+1])
                        if attr not in mentioned_attrs:
                            mentioned_attrs.append(attr)

        qid2hint[qid] = hint_score
    pickle.dump(qid2hint, open(name + '_vqx_hint.pkl', 'wb'))







