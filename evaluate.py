import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
import pickle
from dataset import Dictionary, SelfCriticalDataset
from tqdm import tqdm
from models import Model_explain2
import utils
import opts

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_score_with_k_logits(logits, labels, k=5):
    logits = torch.sort(logits, 1)[1].data  # argmax
    scores = torch.zeros((labels.size(0), k))

    for i in range(k):
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits[:, -i - 1].view(-1, 1), 1)
        scores[:, i] = (one_hots * labels).squeeze().sum(1)
    scores = scores.max(1)[0]
    return scores

def evaluate(model, dataloader):
    score = 0
    scorek = 0
    score1 = 0
    V_loss = 0
    V_loss1 = 0
    qid2type = pickle.load(open('qid2type.pkl', 'rb'))

    upper_bound = 0
    num_data = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for objs, q, a, hintscore, _, qids in tqdm(iter(dataloader)):
        objs = objs.cuda().float().requires_grad_()
        q = q.cuda().long()
        a = a.cuda()  # true labels
        hintscore = hintscore.cuda().float()

        pred, _, ansidx = model(q, objs)

        #loss = instance_bce_with_logits(pred, a)

        #V_loss += loss.item() * objs.size(0)
        batch_score = compute_score_with_logits(pred, a.data).cpu().numpy().sum(1)
        score += batch_score.sum()

        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[qid]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    V_loss /= len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    return score, score_yesno, score_other, score_number


if __name__ == '__main__':
    opt = opts.parse_opt()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    model = Model_explain2(opt)

    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    opt.use_all = 1
    eval_dset = GraphQAIMGDataset('v2cp_test', dictionary, opt)
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=0)

    states_ = torch.load('saved_models/%s/model-best.pth'%opt.load_model_states)
    states = model.state_dict()
    for k in states_.keys():
        if k in states:
            states[k] = states_[k]
            print('copying  %s' % k)
        else:
            print('ignoring  %s' % k)
    model.load_state_dict(states)
    model.eval()
    score, score_yesno, score_other, score_number = evaluate(model, eval_loader)
    print('Overall: %.3f\n' % score)
    print('Yes/No: %.3f\n' % score_yesno)
    print('Number: %.3f\n' % score_number)
    print('Other: %.3f\n' % score_other)

