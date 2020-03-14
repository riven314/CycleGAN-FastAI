import os

from src.data_utils.simulator import MultiDigitSimulator


def build_dataset(data_dir, a_param, b_param, write_dir, train_n, test_n):
    """
    :param:
        a_param : dict, {'is_exists': list, 'affine_tfms': dict}
    """
    # write on train A
    train_a_dir = os.path.join(write_dir, 'trainA')
    os.makedirs(train_a_dir, exist_ok = True)
    simulator = MultiDigitSimulator(
        data_dir, a_param['is_exists'], a_param['affine_tfms'], train_a_dir
        )
    simulator.simulate_n_img(train_n)
    # write on test A
    test_a_dir = os.path.join(write_dir, 'testA')
    os.makedirs(test_a_dir, exist_ok = True)
    simulator.write_dir = test_a_dir
    simulator.simulate_n_img(test_n)
    # write on train B
    simulator.is_exists = b_param['is_exists']
    simulator.affine_tfms = b_param['affine_tfms']
    train_b_dir = os.path.join(write_dir, 'trainB')
    os.makedirs(train_b_dir, exist_ok = True)
    simulator.write_dir = train_b_dir
    simulator.simulate_n_img(train_n)
    # write on test B
    test_b_dir = os.path.join(write_dir, 'testB')
    os.makedirs(test_b_dir, exist_ok = True)
    simulator.write_dir = test_b_dir
    simulator.simulate_n_img(test_n)


if __name__ == '__main__':
    data_dir = os.path.join('data')
    train_n = 12000
    test_n = 200
    # generate medium data
    medium_a_param = {
        'is_exists': [False, True, True, False],
        'affine_tfms': [
            None, {'s': 0.8, 'deg_ccw': 30., 'dx': 5, 'dy': -5},
            {'s': 0.8, 'deg_ccw': 30., 'dx': -5, 'dy': 5}, None
            ]
    }
    medium_b_param = {
        'is_exists': [True, False, False, True],
        'affine_tfms': [
            {'s': 1.15, 'deg_ccw': 0., 'dx': 3, 'dy': 3}, None,
            None, {'s': 1.15, 'deg_ccw': 0., 'dx': -3, 'dy': -3}
            ]
    }
    medium_write_dir = os.path.join('data', 'medium')
    build_dataset(
        data_dir, medium_a_param, medium_b_param, medium_write_dir, train_n, test_n
        )
    # generate easy data
    easy_a_param = {
        'is_exists': [False, False, True, False],
        'affine_tfms': [
            None, None,
            {'s': 0.8, 'deg_ccw': 30., 'dx': -5, 'dy': 5}, None
            ]
    }
    easy_b_param = {
        'is_exists': [False, False, False, True],
        'affine_tfms': [
            None, None,
            None, {'s': 1.15, 'deg_ccw': 0., 'dx': -3, 'dy': -3}
            ]
    }
    easy_write_dir = os.path.join('data', 'easy')
    build_dataset(
        data_dir, easy_a_param, easy_b_param, easy_write_dir, train_n, test_n
        )
