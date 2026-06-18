"""
Main dubbing pipeline — ties all 5 stages together.
Usage:
  from pipeline.dubbing_pipeline import DubbingPipeline
  pipeline = DubbingPipeline()
  output = pipeline.run("path/to/video.mp4")
"""
import os, ffmpeg
from pathlib import Path

from pipeline.transcriber import Transcriber
from pipeline.emotion_detector import EmotionDetector
from pipeline.isochronic import compute_speed_factor
from pipeline.tts_synthesizer import HindiSynthesizer
from pipeline.video_assembler import assemble_dubbed_video, generate_srt

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


class DubbingPipeline:
    def __init__(self, whisper_model='medium', use_pretrained_nmt=True):
        print("Initialising CineSync pipeline...")

        self.transcriber   = Transcriber(model_size=whisper_model)
        self.emotion_det   = EmotionDetector()
        self.synthesizer   = HindiSynthesizer()

        # Use pretrained Helsinki model if custom NMT not trained yet
        if use_pretrained_nmt or not os.path.isfile("models/nmt/nmt_model.pt"):
            from pipeline.translator_pretrained import PretrainedTranslator
            self.translator = PretrainedTranslator()
            print("  Translator: Helsinki-NLP/opus-mt-en-hi (pretrained)")
        else:
            from pipeline.translator import Translator
            self.translator = Translator()
            print("  Translator: Custom BiLSTM NMT")

        print("Pipeline ready.\n")

    def run(self, video_path: str, job_id: str = None, progress_cb=None) -> str:
        """
        Full pipeline: video → Hindi dubbed MP4.
        progress_cb(stage: str, pct: int) called at each stage if provided.
        Returns path to dubbed output file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        job_id = job_id or Path(video_path).stem
        work_dir = OUTPUTS_DIR / job_id
        work_dir.mkdir(exist_ok=True)

        def progress(stage, pct):
            print(f"  [{pct:3d}%] {stage}")
            if progress_cb:
                progress_cb(stage, pct)

        # ── STAGE 1: Extract audio ────────────────────────────────────────────
        progress("Extracting audio", 5)
        audio_path = str(work_dir / "audio.wav")
        (ffmpeg
            .input(video_path)
            .output(audio_path, ar=16000, ac=1, acodec='pcm_s16le')
            .overwrite_output()
            .run(quiet=True))

        # ── STAGE 2: Transcribe ───────────────────────────────────────────────
        progress("Transcribing with Whisper", 15)
        segments = self.transcriber.transcribe(audio_path)
        print(f"         {len(segments)} segments detected")

        if not segments:
            raise ValueError("No speech detected in video.")

        # ── STAGE 3: Translate + Emotion + TTS per segment ───────────────────
        dubbed_segments = []
        n = len(segments)

        for i, seg in enumerate(segments):
            pct = 20 + int((i / n) * 60)
            progress(f"Segment {i+1}/{n}: translate + synthesise", pct)

            # Detect emotion from original audio
            emotion = self.emotion_det.detect(audio_path, seg['start'], seg['end'])

            # Translate English → Hindi
            hindi_text = self.translator.translate(seg['text'])
            if not hindi_text.strip():
                hindi_text = seg['text']   # fallback: keep English

            # Compute speed factor so Hindi fits original timing
            speed = compute_speed_factor(seg['duration'], hindi_text)

            # Synthesise Hindi speech with emotion
            seg_wav = str(work_dir / f"seg_{i:04d}.wav")
            self.synthesizer.synthesize(hindi_text, emotion, seg_wav, speed=speed)

            dubbed_segments.append({
                'start':      seg['start'],
                'end':        seg['end'],
                'audio_path': seg_wav,
                'text':       hindi_text,
                'emotion':    emotion,
            })

            print(f"         [{emotion}] {seg['text'][:40]} → {hindi_text[:40]}")

        # ── STAGE 4: Generate subtitles ───────────────────────────────────────
        progress("Generating subtitles", 82)
        srt_path = str(work_dir / "subtitles.srt")
        generate_srt(dubbed_segments, srt_path)

        # ── STAGE 5: Assemble final video ─────────────────────────────────────
        progress("Assembling dubbed video", 88)
        output_path = str(OUTPUTS_DIR / f"{job_id}_hindi_dubbed.mp4")
        assemble_dubbed_video(video_path, dubbed_segments, output_path)

        # Cleanup segment WAVs
        for seg in dubbed_segments:
            if os.path.isfile(seg['audio_path']):
                os.remove(seg['audio_path'])

        progress("Done", 100)
        print(f"\nOutput: {output_path}")
        return output_path
