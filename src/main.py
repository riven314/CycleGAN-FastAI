import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

from src.model_utils.databunch import *
from src.model_utils.cyclegan import *
from src.model_utils.callbacks import *
from src.model_utils.loss import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'dir to training data (contain trainA, trainB etc.)')
    parser.add_argument('--lr', type = float, default = 3e-4, help = 'learning rate')
    parser.add_argument('--bs', type = int, default = 64, help = 'batch size for training')
    parser.add_argument('--n_epoch', type = int, default = 100, help = 'no. epochs for training')
    parser.add_argument('--img_size', type = int, default = 56, help = 'image size for training (assume square)')
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_name = os.path.basename(data_dir)
    data = (ImageTupleList.from_folders(data_dir, 'trainA', 'trainB')
                          .split_none()
                          .label_empty()
                          .transform(get_transforms(), size = args.img_size)
                          .databunch(bs = args.bs))
    cycle_gan = CycleGAN(3, 3, gen_blocks = 9)
    learn = Learner(data, cycle_gan, 
                    loss_func = CycleGanLoss(cycle_gan), 
                    opt_func = partial(optim.Adam, betas = (0.5,0.99)),
                    callback_fns = [CycleGANTrainer])
    start = time.time()
    learn.fit(args.n_epoch, args.lr)
    learn.save(f'{data_name}_100fit')
    end = time.time()
    print(f'training complete: {(end - start) / 60} mins')
