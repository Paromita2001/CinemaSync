"""
NMT Evaluation — computes BLEU score on test set using beam search.
Usage: python training/evaluate_nmt.py
"""
import os, sys
import torch
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.vocab import tokenize_en, encode, decode, load_iitb_corpus
from models.nmt.seq2seq import build_model

MODEL_PATH = "models/nmt/nmt_model.pt"
VOCAB_PATH = "models/nmt/vocab.pt"
EN_PATH    = "data/iitb_corpus/IITB.en-hi.en"
HI_PATH    = "data/iitb_corpus/IITB.en-hi.hi"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TEST     = 2000   # evaluate on last 2000 pairs


def evaluate():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VOCAB_PATH):
        print("Model not found. Run train_nmt.py first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    vocab_data = torch.load(VOCAB_PATH, map_location=DEVICE)

    src_vocab = vocab_data['src_vocab']
    tgt_vocab = vocab_data['tgt_vocab']
    tgt_inv   = vocab_data['tgt_inv']
    sos_idx   = vocab_data['sos']
    eos_idx   = vocab_data['eos']

    model = build_model(
        checkpoint['src_vocab_size'],
        checkpoint['tgt_vocab_size'],
        DEVICE
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    pairs = load_iitb_corpus(EN_PATH, HI_PATH, max_pairs=999999)
    test_pairs = pairs[-N_TEST:]

    hypotheses, references = [], []

    print(f"Evaluating {N_TEST} sentences with beam search (beam=5)...")
    for i, (en, hi_ref) in enumerate(test_pairs):
        tokens = tokenize_en(en)
        src_ids = encode(tokens, src_vocab)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

        pred_ids = model.translate_beam(src_tensor, sos_idx, eos_idx,
                                         max_len=60, beam_width=5)
        pred_text = decode(pred_ids, tgt_inv)

        hypotheses.append(pred_text)
        references.append(hi_ref)

        if i < 5:
            print(f"\n  EN : {en}")
            print(f"  REF: {hi_ref}")
            print(f"  HYP: {pred_text}")

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_TEST}...")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n{'='*50}")
    print(f"  BLEU Score: {bleu.score:.2f}")
    print(f"  Target:     22.00+")
    print(f"  Status:     {'PASS' if bleu.score >= 22 else 'BELOW TARGET'}")
    print(f"{'='*50}")
    return bleu.score


if __name__ == "__main__":
    evaluate()
