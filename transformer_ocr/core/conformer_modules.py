import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from transformer_ocr.core.attentions import MultiHeadedSelfAttention
from transformer_ocr.core.activations import get_activation_fn


class FeedForward(nn.Module):
    """
    Feed-forward module of Conformer net.
    Args:
        d_model (int):
        d_feedforward (int):
        dropout (float):
        activation:
        layer_norm_eps (float):
    """

    def __init__(self, d_model, d_feedforward, dropout, activation, layer_norm_eps=1e-5):
        super(FeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.activation = activation
        self._dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self._dropout(x)
        x = self.linear2(x)
        x = self._dropout(x)
        return x


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        activation (nn.Module):
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
        expansion_factor (int):
        layer_norm_eps (float):

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            activation,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
            layer_norm_eps=1e-5
    ) -> None:
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels, eps=layer_norm_eps),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels, in_channels * expansion_factor, kernel_size=1),
            get_activation_fn('glu')(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            activation,
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            Transpose(shape=(1, 2)),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sequential(inputs)


class ResidualModule(nn.Module):
    """Residual Connection Module.
    Args:
        module (nn.Module):
        factor (float):

    outputs = module(x) x factor + x
    """
    def __init__(self, module: nn.Module, factor: float = 1.0):
        super(ResidualModule, self).__init__()
        self.module = module
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.module(x) * self.factor) + x


class ConformerEncoderLayer(nn.Module):
    r"""ConformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        n_head: the number of heads in the multiheadattention models (required).
        d_feedforward: the dimension of the feedforward network model (default=2048).
        ff_dropout: the dropout value of Feedforward Module (default=0.1).
        conv_dropout: the dropout value of Convolution Module (default=0.1).
        attn_dropout: the dropout value of SelfAttention Module (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        self_attn_type:
        half_step_residual:
        conv_kernel_size:
        conv_expansion_factor

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, n_head=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, n_head, d_feedforward=2048, ff_dropout=0.1, conv_dropout=0.1, attn_dropout=0.1,
                 activation='swish', layer_norm_eps=1e-5, self_attn_type='abs_pos', half_step_residual=True,
                 conv_kernel_size=31, conv_expansion_factor=2):
        super(ConformerEncoderLayer, self).__init__()

        if self_attn_type == 'abs_pos':
            self.self_attn = MultiHeadedSelfAttention(d_model=d_model,
                                                      num_heads=n_head,
                                                      dropout_p=attn_dropout,
                                                      layer_norm_eps=layer_norm_eps)
        else:
            # TODO
            pass

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

        self.residual_factor = 1.0
        if half_step_residual:
            self.residual_factor = 0.5

        self.net = nn.Sequential(
            ResidualModule(
                module=FeedForward(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=ff_dropout,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps
                ),
                factor=self.residual_factor,
            ),
            ResidualModule(
                module=self.self_attn
            ),
            ResidualModule(
                module=ConvolutionModule(
                    in_channels=d_model,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps
                ),
            ),
            ResidualModule(
                module=FeedForward(
                    d_model=d_model,
                    d_feedforward=d_feedforward,
                    dropout=ff_dropout,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps
                ),
                factor=self.residual_factor,
            ),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)

        return out


def test():
    x = torch.rand((3, 27, 256))
    layer = ConformerEncoderLayer(d_model=256, n_head=4)

    out = layer(x)
    print(out.size())


if __name__ == '__main__':
    test()
