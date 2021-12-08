import torch
from torch import nn
from einops import rearrange
from transformer_ocr.core.embeddings import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Build Transformer Encoder
    Args:
        d_model (int):
        n_head (int):
        n_encoder_layers (int):
        dim_feedforward (int):
        max_seq_length (int):
        pos_dropout (float):
        emb_dropout (float):
        trans_dropout (int):
        scale (bool):
    """
    def __init__(self, vocab_size,
                 d_model,
                 n_head,
                 n_encoder_layers,
                 dim_feedforward,
                 max_seq_length,
                 pos_dropout,
                 emb_dropout,
                 trans_dropout,
                 scale=True):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model=d_model,
                                          dropout=pos_dropout,
                                          dropout_emb=emb_dropout,
                                          max_len=max_seq_length,
                                          scale=scale)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=trans_dropout)
        encoder_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_encoder_layers,
                                                 norm=encoder_norm)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_key_padding_mask=None) -> torch.Tensor:
        src = self.pos_enc(src)
        output = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        output = rearrange(output, 't n c -> n t c')
        output = self.fc(output)
        return output

    def forward_encoder(self, src) -> torch.Tensor:
        src = self.pos_enc(src)
        out = self.transformer(src)
        return out
