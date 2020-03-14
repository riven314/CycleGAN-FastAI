import os

from src.data_utils.simulator import MultiDigitSimulator

data_dir = os.path.join('data')
is_exists = [False, True, True, False]
affine_tfms = [
    None, {'s': 0.8, 'deg_ccw': 40., 'dx': 5, 'dy': -7},
    {'s': 0.7, 'deg_ccw': 30., 'dx': -6, 'dy': 6}, None
    ]
write_dir = os.path.join('data', 'test')
simulator = MultiDigitSimulator(data_dir, is_exists, affine_tfms, write_dir)

random_digit_im = simulator.sample_one_digit_im(digit = None)
fix_digit_im = simulator.sample_one_digit_im(digit = 5)

simulator.simulate_n_img(100)

