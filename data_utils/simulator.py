"""
to be done:
1. support scale, rotation of digit (affine transformation)
"""
import os
import os.path as osp
import argparse
import subprocess

import numpy as np
from imageio import imwrite

from data_utils.downloads import check_mnist_dir, extract_mnist


def sample_coordinate(high, sample_n):
    if high > 0:
        return np.random.randint(high, size = sample_n)
    else:
        return np.zeros(sample_n).astype(np.int)


def simulate_an_image():
    """
    subdivide a simulated image into 4 sqaure regions 
    each region either contains nothing or an affine transformed MNIST 
    each transform takes region's center as origin
    resized to target image size at the end

    :input:
        digits_dict : dict, {digit index: np.array (# samples, W, H, 1)}
        digit_cls_ls : list, class of digit to be shown: [top left, top right, bottom left, bottom right]
                             suppose to be digit index, if None means NO digit in region, 
        digit_tfms_ls : list of affine transformation configuration
        image_size : list, [H, W], resized simulated image into image_size at last
    """
    pass


def generator(config):
    # check if mnist is downloaded. if not, download it
    check_mnist_dir(config.mnist_path)
    # extract mnist images and labels
    image, label = extract_mnist(config.mnist_path)
    h, w = image.shape[1:3]

    # split: train, val, test
    rs = np.random.RandomState(config.random_seed)
    num_original_class = len(np.unique(label))
    # no. of combination = digit_class_n ** num_digit
    num_class = len(np.unique(label)) ** config.num_digit
    classes = list(np.array(range(num_class)))
    rs.shuffle(classes)
    num_train, num_val, num_test = [
            int(float(ratio)/np.sum(config.train_val_test_ratio)*num_class)
            for ratio in config.train_val_test_ratio
            ]
    train_classes = classes[ :num_train]
    val_classes = classes[num_train :num_train + num_val]
    test_classes = classes[num_train+num_val :]

    # label index
    indexes = []
    for c in range(num_original_class):
        indexes.append(list(np.where(label == c)[0]))

    # generate images for every class
    assert config.image_size[1]//config.num_digit >= w # not necessary constraint
    np.random.seed(config.random_seed)

    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)

    split_classes = [train_classes, val_classes, test_classes]
    count = 1
    for i, split_name in enumerate(['train', 'val', 'test']):
        path = osp.join(config.multimnist_path, split_name)
        print(f'generat images for {split_name} at {path}')
        if not os.path.exists(path):
            os.makedirs(path)
        for j, current_class in enumerate(split_classes[i]):
            class_str = f'{current_class:02}'
            class_path = osp.join(path, class_str)
            print(f'{class_path} (progress: {count}/{len(classes)})')
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            for k in range(config.num_image_per_class):
                # sample images
                digits = [int(class_str[l]) for l in range(config.num_digit)]
                imgs = [np.squeeze(image[np.random.choice(indexes[d])]) for d in digits]
                background = np.zeros((config.image_size)).astype(np.uint8)
                # sample coordinates
                ys = sample_coordinate(config.image_size[0] - h, config.num_digit)
                xs = sample_coordinate(config.image_size[1] // config.num_digit - w, config.num_digit)
                xs = [l * config.image_size[1] // config.num_digit + xs[l] for l in range(config.num_digit)]
                # combine images
                for i in range(config.num_digit):
                    background[ys[i]: ys[i] + h, xs[i]: xs[i] + w] = imgs[i]
                # write the image
                image_path = osp.join(class_path, f'{k}_{class_str}.png')
                # image_path = osp.join(config.multimnist_path, '{}_{}_{}.png'.format(split_name, k, class_str))
                imwrite(image_path, background)
            count += 1
    return image, label, indexes


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mnist_path', type = str, default = './datasets/mnist/',
                        help = 'path to *.gz files')
    parser.add_argument('--multimnist_path', type = str, default = './datasets/multimnist')
    parser.add_argument('--num_digit', type = int, default = 2)
    parser.add_argument('--train_val_test_ratio', type = int, nargs = '+',
                        default=[64, 16, 20], help = 'express in percentage e.g. 10% --> 10')
    parser.add_argument('--image_size', type=int, nargs='+',
                        default=[64, 64], help = 'size of simulated images (H, W)') 
    parser.add_argument('--num_image_per_class', type = int, default = 10000)
    parser.add_argument('--random_seed', type = int, default = 123)
    config = parser.parse_args()
    return config


def main():
    config = argparser()
    assert len(config.train_val_test_ratio) == 3
    assert sum(config.train_val_test_ratio) == 100
    assert len(config.image_size) == 2
    generator(config)


if __name__ == '__main__':
    main()