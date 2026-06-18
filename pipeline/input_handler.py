import os, uuid, shutil
import yt_dlp
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}


def download_youtube(url: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    output_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'match_filter': yt_dlp.utils.match_filter_func("duration < 600"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.isfile(output_path):
        # yt-dlp may have added an extension
        candidates = list(UPLOAD_DIR.glob(f"{job_id}.*"))
        if candidates:
            output_path = str(candidates[0])
        else:
            raise FileNotFoundError("YouTube download produced no output file.")

    return output_path


def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported format: {ext}. Allowed: {SUPPORTED_EXTS}")
    job_id = str(uuid.uuid4())[:8]
    output_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    with open(output_path, 'wb') as f:
        f.write(file_bytes)
    return output_path


def validate_youtube_url(url: str) -> dict:
    ydl_opts = {'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'valid': True,
                'title': info.get('title'),
                'duration': info.get('duration'),
                'thumbnail': info.get('thumbnail'),
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}


def is_youtube_url(url: str) -> bool:
    return any(d in url for d in ['youtube.com', 'youtu.be'])
