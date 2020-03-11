import os

from data_utils.geometric import *
from data_utils.downloads import extract_mnist

data_dir = os.path.join('data')
images, labels = extract_mnist(data_dir)
test_np_im = np.squeeze(images[0])

def test_np_to_im():
    im = np_to_im(test_np_im)
    return im


def test_apply_affine_on_img():
    s = 0.5
    dx = 0
    dy = 0
    deg_ccw = 0.
    im = np_to_im(test_np_im)
    trans_im = apply_affine_on_im(im, s, dx, dy, deg_ccw)
    return trans_im


if __name__ == '__main__':
    im = test_np_to_im()
    trans_im = test_apply_affine_on_img()
    trans_im.show()
