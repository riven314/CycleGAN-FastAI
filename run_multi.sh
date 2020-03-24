#!/bin/bash
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/easy --exp_name fit_one_cycle_v2 --n_epoch 40 --is_one_cycle
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/medium --exp_name fit_one_cycle_v2 --n_epoch 40 --is_one_cycle
