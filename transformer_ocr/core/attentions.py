import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import einsum
from einops import rearrange


class MultiHeadedSelfAttention(nn.Module):
    """ MultiHead Attention followed by Vanilla Transformer
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
        layer_norm_eps (float):
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, layer_norm_eps=1e-5):
        super(MultiHeadedSelfAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.layer_norm(x)
        outputs, _ = self.attention(x, x, x, attn_mask=mask)
        outputs = self.dropout(outputs)

        return outputs
