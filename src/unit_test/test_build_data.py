import os

from src.data_utils.build_data import build_dataset

data_dir = os.path.join('data')
a_param = {
    'is_exists': [False, True, True, False],
    'affine_tfms': [
    None, {'s': 0.8, 'deg_ccw': 30., 'dx': 5, 'dy': -5},
    {'s': 0.8, 'deg_ccw': 30., 'dx': -5, 'dy': 5}, None
    ]
}
b_param = {
    'is_exists': [True, False, False, True],
    'affine_tfms': [
    {'s': 1.15, 'deg_ccw': 0., 'dx': 3, 'dy': 3}, None,
    None, {'s': 1.15, 'deg_ccw': 0., 'dx': -3, 'dy': -3}
    ]
}
write_dir = os.path.join('data', 'test')
train_n = 10
test_n = 10

build_dataset(data_dir, a_param, b_param, write_dir, train_n, test_n)