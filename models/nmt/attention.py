import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.
    Scores each encoder output against the current decoder hidden state,
    then returns a weighted context vector.
    """
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # enc_dim*2 because encoder is bidirectional
        self.attn = nn.Linear(enc_dim * 2 + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden:  (1, batch, dec_dim)
        # enc_outputs: (batch, src_len, enc_dim*2)
        src_len = enc_outputs.shape[1]

        dec_hidden = dec_hidden[-1].unsqueeze(1)               # (batch, 1, dec_dim)
        dec_hidden = dec_hidden.repeat(1, src_len, 1)          # (batch, src_len, dec_dim)

        energy = torch.tanh(
            self.attn(torch.cat([dec_hidden, enc_outputs], dim=2))
        )                                                       # (batch, src_len, dec_dim)

        attention = self.v(energy).squeeze(2)                  # (batch, src_len)
        return F.softmax(attention, dim=1)                     # (batch, src_len)
