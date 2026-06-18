import os
import torch
import soundfile as sf
import torchaudio

os.environ["COQUI_TOS_AGREED"] = "1"

# torchaudio 2.6+ requires torchcodec which is not available on CPU/Windows.
# Patch torchaudio.load to use soundfile backend instead.
def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, **kwargs):
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    tensor = torch.from_numpy(data.T if channels_first else data)
    if frame_offset > 0:
        tensor = tensor[..., frame_offset:]
    if num_frames > 0:
        tensor = tensor[..., :num_frames]
    return tensor, sr

torchaudio.load = _sf_load

from TTS.api import TTS

STYLE_REFS = {
    'angry':     'style_refs/angry.wav',
    'sad':       'style_refs/sad.wav',
    'happy':     'style_refs/happy.wav',
    'neutral':   'style_refs/neutral.wav',
    'fearful':   'style_refs/fearful.wav',
    'surprised': 'style_refs/surprised.wav',
}


class HindiSynthesizer:
    def __init__(self):
        print("Loading Coqui XTTS v2...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Verify style reference files exist
        missing = [k for k, v in STYLE_REFS.items() if not os.path.isfile(v)]
        if missing:
            print(f"WARNING: Missing style ref WAVs for: {missing}")
            print("  Run: python training/create_style_refs.py")

    def synthesize(self, hindi_text, emotion, output_path, speed=1.0):
        ref_wav = STYLE_REFS.get(emotion, STYLE_REFS['neutral'])
        if not os.path.isfile(ref_wav):
            ref_wav = next(
                (v for v in STYLE_REFS.values() if os.path.isfile(v)), None
            )
        if ref_wav is None:
            raise FileNotFoundError(
                "No style reference WAVs found in style_refs/. "
                "Run: python training/create_style_refs.py"
            )

        self.tts.tts_to_file(
            text=hindi_text,
            speaker_wav=ref_wav,
            language="hi",
            file_path=output_path,
            speed=speed
        )
        return output_path
