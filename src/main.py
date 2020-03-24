import os
import sys
import time
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch

from src.model_utils.databunch import *
from src.model_utils.cyclegan import *
from src.model_utils.callbacks import *
from src.model_utils.loss import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'dir to training data (contain trainA, trainB, testA, testB)')
    parser.add_argument('--exp_name', type = str, help = 'experiment name')
    parser.add_argument('--lr', type = float, default = 3e-4, help = 'learning rate')
    parser.add_argument('--bs', type = int, default = 64, help = 'batch size for training')
    parser.add_argument('--n_epoch', type = int, default = 100, help = 'no. epochs for training')
    parser.add_argument('--img_size', type = int, default = 56, help = 'image size for training (assume square)')
    parser.add_argument('--is_one_cycle', action = 'store_true', default = False)
    parser.add_argument('--disc_layers', type = int, default = 3, help = 'no. of layers for discriminator A and B')
    parser.add_argument('--gen_blocks', type = int, default = 9, help = 'no. of resnet block for generator A and B')
    parser.add_argument('--loss_iters', type = int, default = 25, help = 'no. of iterations to update loss value in tensorboard')
    parser.add_argument('--hist_iters', type = int, default = 500, help = 'no. of iterations to update histogram value in tensorboard')
    parser.add_argument('--stats_iters', type = int, default = 360, help = 'no of iterations to update model gradients in tensorboard')
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_name = os.path.basename(data_dir)
    data = (ImageTupleList.from_folders(data_dir, 'trainA', 'trainB')
                          .split_none()
                          .label_empty()
                          .transform(get_transforms(), size = args.img_size)
                          .databunch(bs = args.bs))
    cycle_gan = CycleGAN(ch_in = 3, ch_out = 3, 
                         disc_layers = args.disc_layers, 
                         gen_blocks = args.gen_blocks)
    cb = partial(CycleGANTrainer, 
                 base_dir = data_dir, name = args.exp_name,
                 loss_iters = args.loss_iters, 
                 hist_iters = args.hist_iters, 
                 stats_iters = args.stats_iters)
    learn = Learner(data, cycle_gan, 
                    loss_func = CycleGanLoss(cycle_gan), 
                    opt_func = partial(optim.Adam, betas = (0.5, 0.99)),
                    callback_fns = [cb])
    start = time.time()
    if args.is_one_cycle:
        learn.fit_one_cycle(args.n_epoch, args.lr)
        suffix = 'wcycle'
    else:
        learn.fit(args.n_epoch, args.lr)
        suffix = 'wocycle'
    learn.save(f'{data_name}_{suffix}_{args.exp_name}_{args.n_epoch}fit')
    end = time.time()
    print(f'training complete: {(end - start) / 60} mins')
