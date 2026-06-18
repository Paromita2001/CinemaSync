import whisper


class Transcriber:
    def __init__(self, model_size='medium'):
        print(f"Loading Whisper {model_size}...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language='en',
            task='transcribe'
        )
        segments = []
        for seg in result['segments']:
            text = seg['text'].strip()
            if not text:
                continue
            segments.append({
                'text':     text,
                'start':    round(seg['start'], 3),
                'end':      round(seg['end'], 3),
                'duration': round(seg['end'] - seg['start'], 3)
            })
        return segments
