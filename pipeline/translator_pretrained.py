"""
Fallback translator using Helsinki-NLP/opus-mt-en-hi (pretrained, no training needed).
Use this while your custom BiLSTM NMT model is still training.
BLEU ~31 (vs our custom model's target of ~22, but we don't train it ourselves).

Usage:
  from pipeline.translator_pretrained import PretrainedTranslator
  t = PretrainedTranslator()
  print(t.translate("The king is dead."))
"""
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"


class PretrainedTranslator:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(MODEL_NAME)
        self.model.eval()

    def translate(self, english_text):
        inputs = self.tokenizer([english_text], return_tensors="pt",
                                 padding=True, truncation=True, max_length=512)
        translated = self.model.generate(**inputs, num_beams=5, max_length=512)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt",
                                 padding=True, truncation=True, max_length=512)
        translated = self.model.generate(**inputs, num_beams=5, max_length=512)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
