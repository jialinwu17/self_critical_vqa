import pickle
import json
from tqdm import tqdm

ans2label = pickle.load(open('cache/trainval_ans2label.pkl', 'rb'))
label2ans = pickle.load(open('cache/trainval_label2ans.pkl', 'rb'))

qid2q = {}
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
        qid2ans[qid] =  aaa[i]['answer'][j]
    

bbb = pickle.load(open('VQA_caption_traindataset.pkl', 'rb'))
for i in tqdm(range(len(bbb))):
    ents = bbb[i]['question']
    imgid2caption[ents[0]['image_id']] = bbb[i]['caption']
    for j in range(len(ents)):
        ent = ents[j]
        question = ent['question']
        qid = ent['question_id']
        qid2q[qid] = question
        qid2ans[qid] =  bbb[i]['answer'][j]


names = ['train', 'test']
for name in names :
    jsondata = json.load(open('vqacp_v2_%s_annotations.json'%name))
    dataset = []
    imgid2img = {}
    for i in tqdm(range(len(jsondata))):
        ent = jsondata[i]
        imgid = ent['image_id']
        qid = ent['question_id']
        question = qid2q[qid]
        if imgid not in imgid2img:
            imgid2img[imgid] = len(dataset)
            dataset.append({'caption':imgid2caption[imgid], 'question':[], 'answer':[]})
        dataset[imgid2img[imgid]]['question'].append({'image_id':imgid,
             'question': question, 'question_id': qid})
        dataset[imgid2img[imgid]]['answer'].append(qid2ans[qid])
    pickle.dump(dataset, open('VQAcp_caption_%sdataset.pkl'%name, 'wb'))
