import torch
import torch.nn as nn
from models.nmt.attention import BahdanauAttention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(enc_dim, dec_dim)
        # Input = embedding + context vector (enc_dim*2)
        self.rnn = nn.LSTM(
            embed_dim + enc_dim * 2, dec_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        # Output projection: combine rnn output + context + embedding
        self.fc_out = nn.Linear(dec_dim + enc_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, dec_h, dec_c, enc_outputs):
        # tgt_token: (batch,)
        embedded = self.dropout(
            self.embedding(tgt_token.unsqueeze(1))             # (batch, 1, embed_dim)
        )
        attn_w = self.attention(dec_h, enc_outputs)            # (batch, src_len)
        context = torch.bmm(
            attn_w.unsqueeze(1), enc_outputs                   # (batch, 1, enc_dim*2)
        )

        rnn_input = torch.cat([embedded, context], dim=2)     # (batch, 1, embed+enc*2)
        output, (dec_h, dec_c) = self.rnn(rnn_input, (dec_h, dec_c))

        prediction = self.fc_out(
            torch.cat([output, context, embedded], dim=2).squeeze(1)
        )                                                      # (batch, vocab_size)
        return prediction, dec_h, dec_c, attn_w
