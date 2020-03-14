"""
simulate a scene with multi MNIST digit
subdivide a scene by 4 regions (top left, top right, bottom left, bottom right)
each region has the following parameters:
1. if MNIST digit is present in the region
2. if present, specify affine transform parameters to be applied on it
"""
import os
import random

import numpy as np
from PIL import Image

from src.data_utils.geometric import apply_affine_transform, np_to_im, im_to_np
from src.data_utils.downloads import extract_mnist


class MultiDigitSimulator:
    def __init__(self, data_dir, is_exists, affine_tfms, write_dir):
        """
        :param:
            data_dir : str, dir to numpy MNIST digit data
            is_exists : list of 4 value, True or None
            affine_tfms : list of 4 dict, {
                's': float, 'deg_ccw': float, 'dx': int, 'dy': int
                }
        
        * both is_exists, affine_tfms is in order: 
        (top left, top right, bottom left, bottom right)
        """
        self.images, self.labels = self.read_np_data(data_dir)
        self.sample_n, self.w, self.h, _ = self.images.shape
        self.cls_n = self.labels[-1]
        self.is_exists = is_exists
        self.affine_tfms = affine_tfms
        self.write_dir = write_dir
        self.sanity_check()

    def simulate_n_img(self, n):
        for i in range(n):
            sim_im = self.simulate_one_im()
            fname = f'simulated_{i:05}.jpg'
            self.write_im(sim_im, fname)
        print('simulation completed!')

    def simulate_one_im(self):
        sub_im_ls = []
        digit_ls = self.sample_digits()
        print(f'digit_ls: {digit_ls}')
        for digit, tfms in zip(digit_ls, self.affine_tfms):
            if digit is False:
                sub_trans_im = Image.new('L', (self.w, self.h))
            else:
                sub_im = self.sample_one_digit_im(digit = digit)
                sub_trans_im = apply_affine_transform(
                    sub_im, tfms['s'], tfms['deg_ccw'], tfms['dx'], tfms['dy']
                    )
            sub_im_ls.append(sub_trans_im)
        sim_im = Image.new('L', (self.w * 2, self.h * 2))
        # top left, top right, btm left, btm right
        sim_im.paste(sub_im_ls[0], (0, 0)) 
        sim_im.paste(sub_im_ls[1], (self.w, 0))
        sim_im.paste(sub_im_ls[2], (0, self.h))
        sim_im.paste(sub_im_ls[3], (self.w, self.h))
        return sim_im

    def sample_digits(self):
        """ sample digit number for self.is_exists[i] = True """
        digit_ls = []
        for i in self.is_exists:
            digit = i if i is False else random.randint(0, self.cls_n - 1)
            digit_ls.append(digit)
        return digit_ls

    def sample_one_digit_im(self, digit = None):
        """ randomly sample a MNIST digit, unconditional sample if digit = None """
        if digit is None:
            idx = random.randint(0, self.sample_n - 1)
        else:
            assert isinstance(digit, int) and digit <= self.labels.max(), f'wrong arg digit {digit}'
            digit_idxs = np.where(self.labels == digit)[0]
            idx = np.random.choice(digit_idxs, 1)[0]
        np_sample = self.images[idx]
        return np_to_im(np_sample)

    def read_np_data(self, data_dir):
        images, labels = extract_mnist(data_dir)
        return images, labels

    def write_im(self, im, fname):
        w_path = os.path.join(self.write_dir, fname)
        im.save(w_path)
        print(f'image simulated: {w_path}')

    def sanity_check(self):
        assert os.path.isdir(self.write_dir), f'write dir not exist: {self.write_dir}'
        assert self.w == self.h, f'self.w ({self.w}) not match with self.h ({self.h})'
        for digit, tfms in zip(self.is_exists, self.affine_tfms):
            assert (digit is False) == (tfms is None), 'is_exists not aligned with affine_tfms'
        