import os
import numpy as np

from data_utils.geometric import *
from data_utils.downloads import extract_mnist

data_dir = os.path.join('data')
images, labels = extract_mnist(data_dir)
test_np_im = np.squeeze(images[0])


def test_np_to_im():
    im = np_to_im(test_np_im)
    return im


def test_apply_rotate_on_im():
    deg_ccw = 60.
    im = np_to_im(test_np_im)
    # expect anti-clockwise rotation
    trans_im = apply_rotate_on_im(im, deg_ccw)
    return trans_im


def test_apply_translate_on_im():
    dx = 10.
    dy = 10.
    im = np_to_im(test_np_im)
    # expect translate to up left
    trans_im = apply_translate_on_im(im, dx, dy)
    return trans_im


def test_apply_scale_on_im():
    big_im = apply_scale_on_im(im, 1.5)
    small_im = apply_scale_on_im(im, 0.5)
    print(f'big_im.size = {big_im.size}')
    print(f'small_im.size = {small_im.size}')
    return big_im, small_im


def test_apply_affine_transform():
    s = 0.8
    deg_ccw = 60.
    dx = -5.
    dy = 7.
    trans_im = apply_affine_transform(im, s, deg_ccw, dx, dy)
    return trans_im


if __name__ == '__main__':
    im = test_np_to_im()
    trans_im = test_apply_affine_transform()
    trans_im.show()