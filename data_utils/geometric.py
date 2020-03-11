"""
reference
1. doing affine transform with PIL: https://stackoverflow.com/questions/17056209/python-pil-affine-transformation
"""
from PIL import Image
import numpy as np
import math


def np_to_im(np_im):
    return Image.fromarray(np_im, mode = 'L')


def im_to_np(im):
    return np.array(im.convert('L'), dtype = np.uint8)


def apply_affine_on_im(im, s, dx, dy, deg_ccw):
    """
    apply affine transform on input image im
    specified by the following paramters 

    :param:
        im : PIL.Image instance
        s : float, scale applied on im, in both x and y
        dx : int, pixel wise transaltion in x direction applied on im (+ve, shift right)
        dy : int, pixel wise translation in y direction applied on im (+ve, shift down)
        deg_ccw : float, rotation in degree, counter closewise, applied on im
    :output:
        trans_im : PIL.Image, image after affine transform, resized to same size as input
    """
    # define each geometric param
    sx = sy = 1 / s
    w, h = im.size
    angle = math.radians(-deg_ccw)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    scaled_w, scaled_h = w * sx, h * sy
    new_w = int(math.ceil(math.fabs(cos_theta * scaled_w) + math.fabs(sin_theta * scaled_h)))
    new_h = int(math.ceil(math.fabs(sin_theta * scaled_w) + math.fabs(cos_theta * scaled_h)))
    cx = w / 2.
    cy = h / 2.
    tx = new_w / 2.
    ty = new_h / 2.
    # apply adjustment on top of centered image
    tx += dx
    ty += dy

    # calc entries of trans matrix
    a = cos_theta / sx
    b = sin_theta / sx
    c = cx - tx * a - ty * b
    d = -sin_theta / sy
    e = cos_theta / sy
    f = cy - tx * d - ty * e

    # get transformed image
    trans_im = im.transform(
        (new_w, new_h), Image.AFFINE, 
        (a, b, c, d, e, f),
        resample = Image.BILINEAR
        )
    # resized back to original size
    return trans_im.resize((w, h), resample = Image.BILINEAR)