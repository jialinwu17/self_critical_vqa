from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


def set_lr(optimizer, frac):
    for group in optimizer.param_groups:
        group['lr'] *= frac


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(model, train_loader, eval_loader, opt):
    # Paper uses AdaDelta

    if opt.optimizer == 'adadelta':
        optim = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08,
                                    weight_decay=opt.weight_decay, momentum=0, centered=False)
    elif opt.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=opt.weight_decay)


    best_eval_score = 0
    bucket = opt.bucket
    ans_cossim = pickle.load(open('ans_cossim.pkl', 'rb'))
    opt.checkpoint_path = 'saved_models/%s' % (
        str(datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_'))

    os.mkdir(opt.checkpoint_path)
    log_file = open(opt.checkpoint_path + '/log.txt', 'w')
    print(opt, file=log_file)
    log_file.flush()

    if opt.load_hint > 1:
        if 'hatvqx' in opt.split:
            states_ = torch.load('saved_models/2019_05_15_15_18_15_325206/model-best.pth')
        elif 'hat' in opt.split:
            states_ = torch.load('saved_models/2019_05_11_18_51_00_585957/model-best.pth')
        elif 'vqx' in opt.split:
            states_ = torch.load('saved_models/2019_05_13_16_47_03_342162/model-best.pth')
        else:
            states_ = torch.load('saved_models/2019_05_13_22_31_59_650582/model-best.pth')
    elif opt.load_hint > 0:
        states_ = torch.load('saved_models/2019_05_16_13_53_23_519315/model-best.pth')
    else:
        states_ = model.state_dict()

    states = model.state_dict()
    for k in states_.keys():
        if k in states:
            states[k] = states_[k]
            print('copying  %s' % k)
        else:
            print('ignoring  %s' % k)
    model.load_state_dict(states)  
    sigmoid = nn.Sigmoid()
    eps = 0.0000001

    for epoch in range(opt.max_epochs):
        i = 0

        for objs, q, a, hintscore, _, _ in iter(train_loader):
            objs = objs.cuda().float().requires_grad_()
            q = q.cuda().long()
            a = a.cuda()  # true labels
            hintscore = hintscore.cuda().float()

            pred, _, ansidx = model(q, objs)

            loss_vqa = instance_bce_with_logits(pred, a)
            vqa_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), objs, create_graph=True)[0]  # [b , 80, 2048]
            vqa_grad_cam = vqa_grad.sum(2)
            aidx = a.argmax(1).detach().cpu().numpy().reshape((-1))

            loss_hint = torch.zeros((vqa_grad_cam.size(0), opt.num_sub, 36)).cuda()
            hintscore = hintscore.squeeze()
            hint_sort, hint_ind = hintscore.sort(1, descending=True)

            thresh = hint_sort[:, opt.num_sub:opt.num_sub + 1] - 0.00001
            thresh += ((thresh < 0.2).float() * 0.1)
            hintscore = (hintscore > thresh).float()

            for j in range(opt.num_sub):
                for k in range(36):
                    if j == k:
                        continue
                    hint1 = hintscore.gather(1, hint_ind[:, j:j + 1]).squeeze()
                    hint2 = hintscore.gather(1, hint_ind[:, k:k + 1]).squeeze()

                    vqa1 = vqa_grad_cam.gather(1, hint_ind[:, j:j + 1]).squeeze()
                    vqa2 = vqa_grad_cam.gather(1, hint_ind[:, k:k + 1]).squeeze()
                    if j < k:
                        mask = ((hint1 - hint2) * (vqa1 - vqa2 - 0.0001) < 0).float()
                        loss_hint[:, j, k] = torch.abs(vqa1 - vqa2 - 0.0001) * mask
                    else:
                        mask = ((hint2 - hint1) * (vqa2 - vqa1 - 0.0001) < 0).float()
                        loss_hint[:, j, k] = torch.abs(vqa2 - vqa1 - 0.0001) * mask

            loss_hint *= opt.hint_loss_weight
            loss_hint = loss_hint.sum(2)  # b num_sub
            loss_hint += ( ((loss_hint.sum(1).unsqueeze(1) > eps).float() * (loss_hint < eps).float() ) * 10000)

            loss_hint, loss_hint_ind =  loss_hint.min(1) # loss_hint_ind b
            loss_hint_mask = (loss_hint > eps).float()
            loss_hint = (loss_hint * loss_hint_mask).sum() / (loss_hint_mask.sum() + eps)
            logits = pred.gather(1, a.argmax(1).view((-1, 1)))
            prob = sigmoid(logits).view(-1)

            loss_compare = torch.zeros((pred.size(0), bucket)).cuda()
            loss_reg = torch.zeros((pred.size(0), bucket)).cuda()
            comp_mask = torch.zeros((pred.size(0), bucket)).cuda()
            for j in range(bucket):
                logits_pred = pred.gather(1, ansidx[:, j:j + 1])
                prob_pred = sigmoid(logits_pred).squeeze()
                vqa_grad_pred = torch.autograd.grad(pred.gather(1, ansidx[:, j:j + 1]).sum(), objs, create_graph=True)[0]
                vqa_grad_pred_cam = vqa_grad_pred.sum(2)  # b 36
                gradcam_diff = vqa_grad_pred_cam - vqa_grad_cam
                pred_aidx = ansidx[:, j].detach().cpu().numpy().reshape((-1))
                ans_diff = torch.from_numpy(1 - ans_cossim[aidx, pred_aidx].reshape((-1))).cuda().float()
                prob_diff = prob_pred - prob
                prob_diff_relu = prob_diff * (prob_diff > 0).float()

                loss_comp1 = prob_diff_relu.unsqueeze(1) * gradcam_diff * ans_diff.unsqueeze(1) * hintscore
                loss_comp1 = loss_comp1.gather(1, loss_hint_ind.view(-1, 1)).squeeze() #sum(1)
                loss_comp1 *= opt.compare_loss_weight
                loss_compare[:, j] = loss_comp1
                comp_mask[:, j] = (prob_diff > 0).float().squeeze()
                loss_reg[:, j] = (torch.abs(vqa_grad_pred_cam * ans_diff.unsqueeze(1) * (1-hintscore))).sum(1)

            loss_reg = loss_reg.mean() * opt.reg_loss_weight
            #loss_compare = loss_compare.mean()
            loss_compare = (loss_compare * comp_mask).sum() / (comp_mask.sum() + 0.0001)
            loss = loss_vqa + loss_hint + loss_compare + loss_reg
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()
            print("iter %d / %d (epoch %d), vqa = %.3f, hint = %.3f, compare = %.3f, reg = %.3f" % (
                i, len(train_loader), epoch, loss_vqa.item(), loss_hint.item(), loss_compare.item(), loss_reg.item()))

            if i % opt.evaluate_every == 0:
                print("iter %d / %d (epoch %d), vqa = %.3f, hint = %.3f, compare = %.3f, reg = %.3f" % (
                    i, len(train_loader), epoch, loss_vqa.item(), loss_hint.item(), loss_compare.item(),
                    loss_reg.item()), file=log_file)
                log_file.flush()
                model.eval()
                eval_score, bound, V_loss, scorek = evaluate(model, eval_loader)
                print("(epoch %d), eval_score = %.3f, eval_score_k = %.3f" % (epoch, eval_score, scorek))
                print("(epoch %d), eval_score = %.3f, eval_score_k = %.3f" % (epoch, eval_score, scorek), file=log_file)
                log_file.flush()
                model.train()

                if eval_score > best_eval_score:
                    model_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), model_path)
                    best_eval_score = eval_score
            i += 1

        model_path = os.path.join(opt.checkpoint_path, 'model.pth')
        torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    score = 0
    scorek = 0
    V_loss = 0

    upper_bound = 0
    num_data = 0
    for objs, q, a, hintscore, _, _ in tqdm(iter(dataloader)):
        objs = objs.cuda().float().requires_grad_()
        q = q.cuda().long()
        a = a.cuda()  # true labels
        hintscore = hintscore.cuda().float()
        pred, _, ansidx = model(q, objs)
        loss = instance_bce_with_logits(pred, a)
        V_loss += loss.item() * objs.size(0)
        batch_score = compute_score_with_logits(pred, a.data).sum()
        batch_scorek = compute_score_with_k_logits(pred, a.data).sum()
        score += batch_score
        scorek += batch_scorek

        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    scorek = scorek / len(dataloader.dataset)
    V_loss /= len(dataloader.dataset)

    upper_bound = upper_bound / len(dataloader.dataset)

    return score, upper_bound, V_loss, scorek
