import torch
from torch import nn
from timm.models import vgg


class Vgg(nn.Module):
    """ Use variants of VGG-style as a backbone to extract feature from input images. Utilized VGG-networks from
    Timm repo.

    Args:
        name (str): name of pretrained models. For example, vgg11_bn, vgg19_bn, ...
        stride_pool (list): Change kernel_size of MaxPool2d in VGG net to keep more spatial information
        kernel_pool (list): Change stride of MaxPool2d in VGG net to keep more spatial information
        d_model (int): output dimension to feed Transformer model
        pretrained (bool): Whether load pretrained model or not
        dropout_p (float): dropout after extracted feature from CNN backbone
    """
    def __init__(self, name: str,
                 stride_pool: list,
                 kernel_pool: list,
                 d_model: int,
                 pretrained: bool,
                 dropout_p: float):
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

        self.features = net.features
        self.dropout = nn.Dropout(dropout_p)
        self.conv2d = nn.Conv2d(in_channels=512,
                                out_channels=d_model,
                                kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def vgg11_bn(stride_pool, kernel_pool, hidden, pretrained, dropout):
    return Vgg('vgg11_bn', stride_pool, kernel_pool, hidden, pretrained, dropout)


def vgg19_bn(stride_pool, kernel_pool, hidden, pretrained, dropout):
    return Vgg('vgg19_bn', stride_pool, kernel_pool, hidden, pretrained, dropout)


def test():
    x = torch.rand((2, 3, 32, 80))
    model = Vgg('vgg19_bn',
                stride_pool=[[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                kernel_pool=[[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                d_model=144,
                pretrained=True,
                dropout_p=0.1)
    print(model)
    out = model(x)
    print(out.size())


if __name__ == "__main__":
    test()

