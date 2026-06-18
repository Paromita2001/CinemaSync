"""
Phase 3 test — runs the full dubbing pipeline on a test video.
Usage:
  python test_pipeline.py --video path/to/clip.mp4
  python test_pipeline.py --youtube "https://www.youtube.com/watch?v=..."

If no argument given, generates a short English clip using pyttsx3 (offline TTS).
"""
import sys, os, argparse

# Force UTF-8 so Devanagari (Hindi) text can be printed on Windows terminals
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="CineSync pipeline test")
parser.add_argument("--video",   help="Path to a local video file")
parser.add_argument("--youtube", help="YouTube URL to download and dub")
args = parser.parse_args()

# ── Get test video ────────────────────────────────────────────────────────────
if args.video:
    video_path = args.video

elif args.youtube:
    from pipeline.input_handler import download_youtube
    print(f"Downloading: {args.youtube}")
    video_path = download_youtube(args.youtube)
    print(f"Downloaded:  {video_path}")

else:
    # Generate a short English test clip using Windows offline TTS (pyttsx3)
    import pyttsx3, wave, struct
    os.makedirs("uploads", exist_ok=True)
    video_path = "uploads/test_clip.wav"

    if not os.path.isfile(video_path):
        print("Generating English test audio with pyttsx3...")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        test_text = (
            "Hello, welcome to CineSync. "
            "This is a test of the dubbing pipeline. "
            "I am very happy to be here today. "
            "The weather is beautiful outside. "
            "This system can translate English speech into Hindi."
        )
        engine.save_to_file(test_text, video_path)
        engine.runAndWait()
        print(f"Saved: {video_path}")
    else:
        print(f"Using existing: {video_path}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  CineSync — Full Pipeline Test")
print("="*55 + "\n")

from pipeline.dubbing_pipeline import DubbingPipeline

pipeline = DubbingPipeline(whisper_model='small', use_pretrained_nmt=True)
output = pipeline.run(video_path, job_id="pipeline_test")

print("\n" + "="*55)
print(f"  SUCCESS: {output}")
print("="*55)
