import random
import torch
import torch.nn as nn
from models.nmt.encoder import Encoder
from models.nmt.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        enc_outputs, h, c = self.encoder(src)
        # Encoder compresses to 1 layer; repeat to match decoder's n_layers
        n = self.decoder.rnn.num_layers
        h = h.repeat(n, 1, 1)
        c = c.repeat(n, 1, 1)

        # First decoder input is <sos> token
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            pred, h, c, _ = self.decoder(dec_input, h, c, enc_outputs)
            outputs[:, t] = pred
            # Teacher forcing: use real target or model prediction
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else pred.argmax(1)

        return outputs

    def translate_greedy(self, src_tensor, sos_idx, eos_idx, max_len=50):
        self.eval()
        with torch.no_grad():
            enc_outputs, h, c = self.encoder(src_tensor)
            n = self.decoder.rnn.num_layers
            h = h.repeat(n, 1, 1)
            c = c.repeat(n, 1, 1)
            dec_input = torch.tensor([sos_idx]).to(self.device)
            tokens = []
            for _ in range(max_len):
                pred, h, c, _ = self.decoder(dec_input, h, c, enc_outputs)
                top = pred.argmax(1).item()
                if top == eos_idx:
                    break
                tokens.append(top)
                dec_input = torch.tensor([top]).to(self.device)
        return tokens

    def translate_beam(self, src_tensor, sos_idx, eos_idx,
                       max_len=50, beam_width=5):
        self.eval()
        with torch.no_grad():
            enc_outputs, h, c = self.encoder(src_tensor)
            n = self.decoder.rnn.num_layers
            h = h.repeat(n, 1, 1)
            c = c.repeat(n, 1, 1)

        beams = [(0.0, [sos_idx], h, c)]
        completed = []

        import torch.nn.functional as F
        for _ in range(max_len):
            new_beams = []
            for score, toks, bh, bc in beams:
                dec_input = torch.tensor([toks[-1]]).to(self.device)
                with torch.no_grad():
                    pred, bh, bc, _ = self.decoder(dec_input, bh, bc, enc_outputs)
                log_probs = F.log_softmax(pred, dim=-1)
                top = log_probs.topk(beam_width)
                for lp, tok in zip(top.values[0], top.indices[0]):
                    new_score = score + lp.item()
                    new_toks = toks + [tok.item()]
                    if tok.item() == eos_idx:
                        completed.append((new_score / len(new_toks), new_toks))
                    else:
                        new_beams.append((new_score, new_toks, bh, bc))
            if not new_beams:
                break
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        if not completed:
            completed = [(b[0], b[1]) for b in beams]

        best = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
        return best[1:]  # strip <sos>


def build_model(src_vocab_size, tgt_vocab_size, device,
                embed_dim=256, hidden_dim=512, n_layers=2, dropout=0.3):
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, hidden_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)
    return model
