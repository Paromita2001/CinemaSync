"""
Picks one clean audio clip per emotion from RAVDESS and copies it to style_refs/.
These are used by Coqui XTTS v2 as voice style references.
Usage: python training/create_style_refs.py
Run AFTER downloading RAVDESS (python training/download_data.py)
"""
import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAVDESS_DIR = "data/ravdess"
STYLE_DIR   = "style_refs"

# RAVDESS emotion code → target emotion name
# Format: Actor_XX/03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
# We want intensity=02 (strong) for clearer emotion signal
EMOTION_TARGETS = {
    'neutral':   '01',
    'happy':     '03',
    'sad':       '04',
    'angry':     '05',
    'fearful':   '06',
    'surprised': '08',
}


def find_clip(emotion_code, intensity='02', fallback_intensity='01'):
    for actor_dir in sorted(os.listdir(RAVDESS_DIR)):
        actor_path = os.path.join(RAVDESS_DIR, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for f in sorted(os.listdir(actor_path)):
            if not f.endswith('.wav'):
                continue
            parts = f.replace('.wav', '').split('-')
            if len(parts) < 7:
                continue
            if parts[2] == emotion_code and parts[3] == intensity:
                return os.path.join(actor_path, f)
    # Fallback to normal intensity
    if fallback_intensity and fallback_intensity != intensity:
        return find_clip(emotion_code, fallback_intensity, fallback_intensity=None)
    return None


def create_refs():
    if not os.path.isdir(RAVDESS_DIR) or not os.listdir(RAVDESS_DIR):
        print("RAVDESS not found. Run: python training/download_data.py first")
        return

    os.makedirs(STYLE_DIR, exist_ok=True)
    print("Creating style reference WAVs from RAVDESS...\n")

    for emotion, code in EMOTION_TARGETS.items():
        clip = find_clip(code)
        if clip is None:
            print(f"  [!!] {emotion:<12} — no clip found (code {code})")
            continue
        dest = os.path.join(STYLE_DIR, f"{emotion}.wav")
        shutil.copy2(clip, dest)
        print(f"  [OK] {emotion:<12} → {dest}  (src: {os.path.basename(clip)})")

    print(f"\nDone. style_refs/ now has {len(os.listdir(STYLE_DIR))} WAV files.")
    print("These will be used by Coqui XTTS v2 for emotion-conditioned synthesis.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    create_refs()
