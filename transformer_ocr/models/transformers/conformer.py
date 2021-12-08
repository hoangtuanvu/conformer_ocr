import torch
import torch.nn as nn
from einops import rearrange

from transformer_ocr.core.embeddings import PositionalEncoding
from transformer_ocr.core.conformer_modules import ConformerEncoderLayer


class ConformerEncoder(nn.Module):
    def __init__(self, vocab_size,
                 max_seq_length,
                 n_layers,
                 scale,
                 d_model,
                 n_head,
                 d_feedforward,
                 emb_dropout,
                 pos_dropout,
                 ff_dropout,
                 conv_dropout,
                 attn_dropout,
                 activation,
                 layer_norm_eps,
                 self_attn_type,
                 half_step_residual,
                 conv_kernel_size,
                 conv_expansion_factor):
        super(ConformerEncoder, self).__init__()

        print('vocab_size', vocab_size)
        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model=d_model,
                                          dropout=pos_dropout,
                                          dropout_emb=emb_dropout,
                                          max_len=max_seq_length,
                                          scale=scale)

        self.layers = nn.ModuleList([ConformerEncoderLayer(
            d_model,
            n_head,
            d_feedforward,
            ff_dropout,
            conv_dropout,
            attn_dropout,
            activation,
            layer_norm_eps,
            self_attn_type,
            half_step_residual,
            conv_kernel_size,
            conv_expansion_factor
        ) for _ in range(n_layers)])

        self.fc = nn.Linear(d_model, vocab_size)
        self.reset_parameters()

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        output = self.pos_enc(output)

        for layer in self.layers:
            output = layer(output)

        output = output.transpose(0, 1)
        # output = rearrange(output, 't n c -> n t c')
        output = self.fc(output)
        return output

    def reset_parameters(self):
        if self.fc is not None:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.)



