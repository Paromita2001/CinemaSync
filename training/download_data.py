"""
Auto-download RAVDESS and CREMA-D datasets.
IIT Bombay corpus requires manual download (see DOWNLOAD_DATASETS.txt).
Usage: python training/download_data.py
"""
import os, sys, zipfile, subprocess, urllib.request

RAVDESS_URL  = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
RAVDESS_ZIP  = "data/ravdess/Audio_Speech_Actors_01-24.zip"
RAVDESS_DIR  = "data/ravdess"

CREMA_DIR    = "data/crema_d"
CREMA_REPO   = "https://github.com/CheyneyComputerScience/CREMA-D.git"


def download_ravdess():
    if os.path.isfile(os.path.join(RAVDESS_DIR, "Actor_01", "03-01-01-01-01-01-01.wav")):
        print("RAVDESS already downloaded.")
        return True

    print("Downloading RAVDESS (~750 MB)...")
    try:
        def progress(count, block_size, total_size):
            pct = min(count * block_size * 100 // total_size, 100)
            print(f"\r  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(RAVDESS_URL, RAVDESS_ZIP, reporthook=progress)
        print()

        print("Extracting...")
        with zipfile.ZipFile(RAVDESS_ZIP, 'r') as zf:
            zf.extractall(RAVDESS_DIR)
        os.remove(RAVDESS_ZIP)
        print("RAVDESS ready.")
        return True
    except Exception as e:
        print(f"RAVDESS download failed: {e}")
        return False


def download_crema():
    audio_dir = os.path.join(CREMA_DIR, "AudioWAV")
    if os.path.isdir(audio_dir) and len(os.listdir(audio_dir)) > 100:
        print("CREMA-D already downloaded.")
        return True

    print("Cloning CREMA-D (~2 GB via git)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", CREMA_REPO, CREMA_DIR],
            check=True
        )
        print("CREMA-D ready.")
        return True
    except Exception as e:
        print(f"CREMA-D clone failed: {e}")
        print("Manual fix: git clone https://github.com/CheyneyComputerScience/CREMA-D data/crema_d")
        return False


def check_iitb():
    en_path = "data/iitb_corpus/IITB.en-hi.en"
    hi_path = "data/iitb_corpus/IITB.en-hi.hi"
    if os.path.isfile(en_path) and os.path.isfile(hi_path):
        with open(en_path) as f:
            lines = sum(1 for _ in f)
        print(f"IIT Bombay corpus ready ({lines:,} lines).")
        return True
    else:
        print("IIT Bombay corpus NOT found.")
        print("  Manual download required:")
        print("  1. Go to http://www.cfilt.iitb.ac.in/iitb_parallel/")
        print("  2. Download iitb.en-hi.tar.gz")
        print("  3. Extract to data/iitb_corpus/")
        print("  Expected files:")
        print("    data/iitb_corpus/IITB.en-hi.en")
        print("    data/iitb_corpus/IITB.en-hi.hi")
        return False


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("=== CineSync Dataset Downloader ===\n")

    r = download_ravdess()
    print()
    c = download_crema()
    print()
    i = check_iitb()

    print("\n=== Summary ===")
    print(f"  RAVDESS  : {'OK' if r else 'FAILED'}")
    print(f"  CREMA-D  : {'OK' if c else 'FAILED'}")
    print(f"  IIT Bombay: {'OK' if i else 'MANUAL DOWNLOAD NEEDED'}")

    if r and c:
        print("\nEmotion training data ready.")
        print("Run: python training/train_emotion.py")
    if i:
        print("NMT training data ready.")
        print("Run: python training/train_nmt.py")
