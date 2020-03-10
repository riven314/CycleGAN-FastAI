"""
reference:
1. how to import module in unit test: https://stackoverflow.com/questions/51744057/how-to-import-module-into-a-python-unit-test
"""
import os

from data_utils.downloads import *

data_dir = os.path.join('data')

def test_checkmnist_dir():
    check_mnist_dir(data_dir)


def test_extract_mnist():
    train_data, test_data = extract_mnist(data_dir)
    return train_data, test_data


if __name__ == '__main__':
    test_checkmnist_dir()
    train_data, test_data = test_extract_mnist()