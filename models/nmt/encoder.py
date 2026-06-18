import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        # Compress bidirectional (hidden_dim*2) → hidden_dim for decoder
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))           # (batch, src_len, embed_dim)
        outputs, (h, c) = self.rnn(embedded)
        # outputs: (batch, src_len, hidden_dim*2)
        # h/c:     (n_layers*2, batch, hidden_dim)

        # Merge forward + backward final hidden states
        h = torch.tanh(self.fc_h(
            torch.cat([h[-2], h[-1]], dim=1)                  # (batch, hidden_dim*2)
        )).unsqueeze(0)                                        # (1, batch, hidden_dim)

        c = torch.tanh(self.fc_c(
            torch.cat([c[-2], c[-1]], dim=1)
        )).unsqueeze(0)

        return outputs, h, c
