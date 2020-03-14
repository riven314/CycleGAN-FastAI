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
