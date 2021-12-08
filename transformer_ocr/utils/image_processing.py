import math
import numpy as np
from PIL import Image


def get_new_width(old_w, old_h, expected_height, image_min_width, image_max_width):
    """TODO
    """
    new_w = int(expected_height * float(old_w) / float(old_h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w


def resize_img(image, image_height, image_min_width, image_max_width):
    """TODO
    """
    img = image.convert('RGB')

    old_w, old_h = img.size
    new_w = get_new_width(old_w, old_h, image_height, image_min_width, image_max_width)
    new_im = img.resize((new_w, image_height), Image.BICUBIC)

    new_im = np.asarray(new_im, dtype=np.float32).transpose(2, 0, 1)
    new_im = new_im / 255
    return new_im

