"""
reference:
1. how to import module in unit test: https://stackoverflow.com/questions/51744057/how-to-import-module-into-a-python-unit-test
"""
import os

from src.data_utils.downloads import *

data_dir = os.path.join('data')


def test_checkmnist_dir():
    check_mnist_dir(data_dir)


def test_extract_mnist():
    images, labels = extract_mnist(data_dir)
    return images, labels


if __name__ == '__main__':
    test_checkmnist_dir()
    images, labels = test_extract_mnist()