import os
import sys
from pathlib import Path

import numpy as np
import torch

from data_utils import *
from models_utils import *
from callbacks_utils import *
from loss_utils import *


if __name__ == '__main__':
    path = Path('../data/horse2zebra/horse2zebra')
    data = (ImageTupleList.from_folders(path, 'trainA', 'trainB')
                      .split_none()
                      .label_empty()
                      .transform(get_transforms(), size=128)
                      .databunch(bs=4))
    cycle_gan = CycleGAN(3,3, gen_blocks=9)
    learn = Learner(data, cycle_gan, 
                    loss_func=CycleGanLoss(cycle_gan), 
                    opt_func=partial(optim.Adam, betas=(0.5,0.99)),
                    callback_fns=[CycleGANTrainer])
    learn.fit(100, 1e-4)
    learn.save('cyclegan_100fit')