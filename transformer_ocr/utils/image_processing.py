import torch
import math
import torchvision.transforms as transforms
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

    return new_im


class NormalizePAD(object):
    """TODO
    """
    def __init__(self, max_size):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)

    def __call__(self, img):
        img = self.toTensor(img)
        img = img / 255
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class ResizeNormalize(object):
    """TODO
    """
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img