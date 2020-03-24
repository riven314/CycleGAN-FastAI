#!/bin/bash
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/easy --exp_name fit_one_cycle_2_period_500_warmup --n_epoch 40 --is_one_cycle --critic_period 2 --warmup_iters 500
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/medium --exp_name fit_one_cycle_2_period_500_warmup --n_epoch 40 --is_one_cycle --critc_period 2 --warmup_iters 500

python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/easy --exp_name fit_one_cycle_2_period --n_epoch 40 --is_one_cycle --critic_period 2
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/medium --exp_name fit_one_cycle_2_period --n_epoch 40 --is_one_cycle --critc_period 2

python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/easy --exp_name fit_one_cycle --n_epoch 40 --is_one_cycle
python -m src.main --data_dir /userhome/34/h3509807/fastai/CycleGAN-MultiMNIST/data/medium --exp_name fit_one_cycle --n_epoch 40 --is_one_cycle


