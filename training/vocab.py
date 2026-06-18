"""Vocabulary builder for NMT — builds word-to-index mappings from corpus."""
import re
from collections import Counter


SPECIAL = ['<pad>', '<unk>', '<sos>', '<eos>']
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def tokenize_en(text):
    return re.findall(r"[a-zA-Z']+|[.,!?;]", text.lower())


def tokenize_hi(text):
    # Split on whitespace and punctuation for Devanagari
    return re.split(r'(\s+|[।,!?;])', text.strip())


def build_vocab(sentences, tokenize_fn, max_vocab=20000, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize_fn(sent))

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL)}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]


def decode(indices, vocab_inv, stop_at_eos=True):
    tokens = []
    for idx in indices:
        if stop_at_eos and idx == EOS_IDX:
            break
        if idx not in (PAD_IDX, SOS_IDX):
            tokens.append(vocab_inv.get(idx, '<unk>'))
    return ' '.join(tokens)


def load_iitb_corpus(en_path, hi_path, max_pairs=300000):
    with open(en_path, encoding='utf-8') as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(hi_path, encoding='utf-8') as f:
        hi_lines = [l.strip() for l in f if l.strip()]

    pairs = list(zip(en_lines, hi_lines))[:max_pairs]
    # Filter out very long sentences (slow to train and rare)
    pairs = [(e, h) for e, h in pairs if len(e.split()) <= 50 and len(h.split()) <= 50]
    return pairs
