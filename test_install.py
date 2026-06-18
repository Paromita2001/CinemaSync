"""
Run this to verify all Phase 1 dependencies are installed.
Usage: python test_install.py
"""
import sys

results = []

def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except Exception as e:
        results.append((name, False, str(e)[:80]))

check("whisper",       lambda: __import__("whisper"))
check("torch",         lambda: __import__("torch"))
check("torchaudio",    lambda: __import__("torchaudio"))
check("tensorflow",    lambda: __import__("tensorflow"))
check("librosa",       lambda: __import__("librosa"))
check("soundfile",     lambda: __import__("soundfile"))
check("pydub",         lambda: __import__("pydub"))
check("ffmpeg",        lambda: __import__("ffmpeg"))
check("yt_dlp",        lambda: __import__("yt_dlp"))
check("fastapi",       lambda: __import__("fastapi"))
check("uvicorn",       lambda: __import__("uvicorn"))
check("numpy",         lambda: __import__("numpy"))
check("pandas",        lambda: __import__("pandas"))
check("sklearn",       lambda: __import__("sklearn"))
check("matplotlib",    lambda: __import__("matplotlib"))
check("streamlit",     lambda: __import__("streamlit"))
check("sacrebleu",     lambda: __import__("sacrebleu"))
check("pytest",        lambda: __import__("pytest"))
check("indicnlp",      lambda: __import__("indicnlp"))

# TTS is optional - requires Visual C++ Build Tools on Windows
try:
    from TTS.api import TTS
    results.append(("TTS (Coqui)", True, ""))
except Exception as e:
    results.append(("TTS (Coqui)", False, "OPTIONAL - needs Visual C++ Build Tools on Windows"))

# Check FFmpeg system binary
import subprocess
try:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if out.returncode == 0:
        version = out.stdout.split('\n')[0]
        results.append(("FFmpeg (system)", True, version[:60]))
    else:
        results.append(("FFmpeg (system)", False, "not found in PATH"))
except Exception as e:
    results.append(("FFmpeg (system)", False, str(e)))

# Summary
print("\n" + "="*60)
print("  CineSync Phase 1 — Installation Check")
print("="*60)
for name, ok, note in results:
    status = "[OK]" if ok else "[!!]"
    print(f"  {status}  {name:<22} {note if not ok else ''}")

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print("="*60)
print(f"  {passed}/{total} checks passed")

import torch
print(f"\n  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.cuda.is_available()}")
print(f"  Python   : {sys.version.split()[0]}")
