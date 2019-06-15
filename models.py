import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import timeit

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding('data/glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([2048, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=2048, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=2048, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=3129,
                                           dropout=dropC, norm=norm, act=activation)

    def forward(self, q, gv):

        """Forward
        q: [batch_size, seq_length]
        c: [batch, 5, 20]
        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]

        att_1 = self.gv_att_1(gv, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(gv, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2
        gv_embs = (att_gv * gv)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        q_repr = self.q_net(q_emb)
        gv_repr = self.gv_net(gv_emb)
        joint_repr = q_repr * gv_repr

        logits = self.classifier(joint_repr)
        ansidx = torch.argsort(logits, dim=1, descending=True)

        return logits, att_gv,  ansidx
