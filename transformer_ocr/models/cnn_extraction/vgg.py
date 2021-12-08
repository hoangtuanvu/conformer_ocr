import torch
from torch import nn
from torchvision.models import vgg
# from timm.models import vgg
from transformer_ocr.core.activations import Swish


class Vgg(nn.Module):
    """
    Args:
        name (str):
        stride_pool (list):
        kernel_pool (list):
        d_model (int):
        pretrained (bool):
        dropout_p (float):
    """
    def __init__(self, name, stride_pool, kernel_pool, d_model, pretrained=True, dropout_p=0.5):
        super(Vgg, self).__init__()

        assert name in ['vgg11_bn', 'vgg19_bn'], "{} does not in the pre-defined list".format(name)

        if name == 'vgg11_bn':
            net = vgg.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            net = vgg.vgg19_bn(pretrained=pretrained)

        pool_idx = 0

        for i, layer in enumerate(net.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                net.features[i] = torch.nn.AvgPool2d(kernel_size=list(kernel_pool[pool_idx]),
                                                     stride=list(stride_pool[pool_idx]),
                                                     padding=0)
                pool_idx += 1
            # elif isinstance(layer, torch.nn.ReLU):
            #     net.features[i] = Swish()

        self.features = net.features
        self.dropout = nn.Dropout(dropout_p)
        self.conv2d = nn.Conv2d(in_channels=512,
                                out_channels=d_model,
                                kernel_size=1)

    def forward(self, x):
        """
        Shape:
            - x: (n, C, H, W)
            - output: (t, n, c)
        """
        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.conv2d(conv)

        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv


def vgg11_bn(stride_pool, kernel_pool, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg11_bn', stride_pool, kernel_pool, hidden, pretrained, dropout)


def vgg19_bn(stride_pool, kernel_pool, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', stride_pool, kernel_pool, hidden, pretrained, dropout)


if __name__ == "__main__":
    x = torch.rand((2, 3, 16, 32))
    model = Vgg('vgg19_bn',
                stride_pool=[[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                kernel_pool=[[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                d_model = 144,
                pretrained=True)
    print(model)
    out = model(x)
    print(out.size())

