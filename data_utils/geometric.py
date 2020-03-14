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


def apply_affine_transform(im, s, deg_ccw, dx, dy):
    """
    apply in order: (i) scaling, (ii) rotation, (iii) translation

    :param:
        im : PIL.Image instance
        s : float, enlarge/ shrink (>1 / <1)
        deg_ccw : float, rotation in degree, counter closewise, applied on im
        dx : int, pixel wise transaltion on left, right (+ve/ -ve)
        dy : int, pixel wise translation on up, down (+ve/ -ve)

    :output:
        trans_im : PIL.Image, image after affine transform, resized to same size as input
    """
    trans_im = apply_scale_on_im(im, s)
    trans_im = apply_rotate_on_im(trans_im, deg_ccw)
    trans_im = apply_translate_on_im(trans_im, dx, dy)
    return trans_im


def apply_scale_on_im(im, s):
    """
    apply scaling on im, assume same scale applied on both x, y axis

    :param:
        im : PIL.Image instance
        s : float, enlarge/ shrink (>1 / <1)
    :output:
        trans_im : PIL.Image, image after affine transform, resized to same size as input
    """
    w, h = im.size
    # pad image if shrink 
    if s < 1:
        new_w, new_h = int(w / s), int(h / s)
        trans_im = Image.new("L", (new_w, new_h))
        trans_im.paste(im, ((new_w - w) // 2, (new_h - h) // 2))
    # crop image on center if enlarge
    elif s > 1:
        # left, upper, right, and lower
        new_w, new_h = w / s, h / s
        left = (w - new_w) // 2
        right = left + new_w
        top = (h - new_h) // 2
        bottom = top + new_h
        trans_im = im.crop((left, top, right, bottom))
    else:
        trans_im = im
    return trans_im.resize(size = (w, h), resample = Image.BILINEAR)


def apply_translate_on_im(im, dx, dy):
    """
    :param:
        im : PIL.Image instance
        dx : int, pixel wise transaltion on left, right (+ve/ -ve)
        dy : int, pixel wise translation on up, down (+ve/ -ve)
    :output:
        trans_im : PIL.Image, image after affine transform, resized to same size as input
    """
    return im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))


def apply_rotate_on_im(im, deg_ccw):
    """
    apply rotation on image center (i.e. origin at w/2, h/2)

    :param:
        im : PIL.Image instance
        deg_ccw : float, rotation in degree, counter closewise, applied on im
    :output:
        trans_im : PIL.Image, image after affine transform, resized to same size as input
    """
    # define each geometric param
    sx = sy = 1.
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