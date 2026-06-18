import os
import torch

from training.vocab import tokenize_en, encode, decode
from models.nmt.seq2seq import build_model

MODEL_PATH = "models/nmt/nmt_model.pt"
VOCAB_PATH = "models/nmt/vocab.pt"


class Translator:
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')

        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VOCAB_PATH):
            raise FileNotFoundError(
                "NMT model not found. Run training/train_nmt.py first."
            )

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        vocab_data = torch.load(VOCAB_PATH, map_location=self.device)

        self.src_vocab = vocab_data['src_vocab']
        self.tgt_inv   = vocab_data['tgt_inv']
        self.sos_idx   = vocab_data['sos']
        self.eos_idx   = vocab_data['eos']

        self.model = build_model(
            checkpoint['src_vocab_size'],
            checkpoint['tgt_vocab_size'],
            self.device
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def translate(self, english_text, beam_width=5):
        tokens = tokenize_en(english_text)
        src_ids = encode(tokens, self.src_vocab)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)

        pred_ids = self.model.translate_beam(
            src_tensor, self.sos_idx, self.eos_idx,
            max_len=100, beam_width=beam_width
        )
        return decode(pred_ids, self.tgt_inv)
