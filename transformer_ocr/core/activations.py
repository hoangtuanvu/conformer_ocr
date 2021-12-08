import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """
    Applies the gated linear unit function introduced in 'https://arxiv.org/abs/1612.08083v3'
    with 2-time duplications of input”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return nn.SiLU()
    elif activation == 'glu':
        return GLU

    raise RuntimeError("activation should be relu/gelu/swish, not {}".format(activation))