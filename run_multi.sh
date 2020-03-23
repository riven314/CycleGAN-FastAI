#!/bin/bash
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/easy --exp_name fit_one_cycle_v1 --n_epoch 40
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/medium --exp_name fit_one_cycle_v1 --n_epoch 40
