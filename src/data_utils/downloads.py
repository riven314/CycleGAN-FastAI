"""
util for downloading required datasets. those datasets serve as raw materials for simulation
"""
import os
import os.path as osp
import gzip
import shutil
import subprocess

import numpy as np

data_url = 'http://yann.lecun.com/exdb/mnist/{}'
mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']


def check_mnist_dir(data_dir):
    downloaded = np.all([osp.isfile(osp.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        print('MNIST not found, downloading them...')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST found')


def download_mnist(data_dir):
    for unzip_f in mnist_keys:
        zip_f = unzip_f + '.gz'
        url = data_url.format(zip_f)
        src_path = os.path.join(data_dir, zip_f)
        tgt_path = os.path.join(data_dir, unzip_f)
        cmd = ['curl', url, '-o', src_path]
        print('Downloading ', zip_f)
        subprocess.call(cmd)
        unzip_file(src_path, tgt_path)


def unzip_file(src_path, tgt_path):
    with gzip.open(src_path, 'rb') as f_in:
        with open(tgt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def extract_mnist(data_dir):
    """
    :return:
        images : np.array (70000, 28, 28, 1)
        labels : np.array (70000, )
    """
    train_n = 60000
    test_n = 10000
    train_image = load_np_objs(data_dir, 'train-images-idx3-ubyte', n = train_n, np_type = 'image')
    train_label = load_np_objs(data_dir, 'train-labels-idx1-ubyte', n = train_n, np_type = 'label')
    test_image = load_np_objs(data_dir, 't10k-images-idx3-ubyte', n = test_n, np_type = 'image')
    test_label = load_np_objs(data_dir, 't10k-labels-idx1-ubyte', n = test_n, np_type = 'label')
    return np.concatenate((train_image, test_image)), np.concatenate((train_label, test_label))


def load_np_objs(data_dir, mnist_key, n, np_type):
    read_path = os.path.join(data_dir, mnist_key)
    assert np_type in ['image', 'label'], f'wrong arg np_type: {np_type}'
    assert os.path.isfile(read_path), f'file not exist: {read_path}'
    fd = open(read_path)
    loaded = np.fromfile(file = fd, dtype = np.uint8)
    if np_type == 'image':
        return loaded[16:].reshape((n, 28, 28, 1))
    else:
        return np.asarray(loaded[8:].reshape((n)))


if __name__ == '__main__':
    pass