import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from dataset_v2 import Dictionary, SelfCriticalDataset
from models import Model
import utils
import opts
from train import train

def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)

if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(args.seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    
    model = Model(opt)
    model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()

    train_dset = SelfCriticalDataset(opt.split, dictionary, opt)
    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=0)
    opt.use_all = 1
    eval_dset = SelfCriticalDataset(opt.split_test, dictionary, opt)
    eval_loader  = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=0)

    train(model, train_loader, eval_loader, opt)
