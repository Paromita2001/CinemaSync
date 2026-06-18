import os
import ffmpeg
from pydub import AudioSegment


def assemble_dubbed_video(original_video, dubbed_segments, output_path, subtitle_path=None):
    """
    Overlays Hindi audio segments onto the original video at their timestamps.
    dubbed_segments: list of {'start': float, 'audio_path': str, 'text': str}
    If the input has no video stream (e.g. a WAV file), outputs dubbed audio only.
    """
    probe = ffmpeg.probe(original_video)
    duration_ms = int(float(probe['format']['duration']) * 1000)

    # Check whether the source has a video stream
    has_video = any(s['codec_type'] == 'video' for s in probe['streams'])

    # Build full-length silent audio track, then overlay each Hindi segment
    final_audio = AudioSegment.silent(duration=duration_ms)
    for seg in dubbed_segments:
        if not os.path.isfile(seg['audio_path']):
            continue
        hindi_audio = AudioSegment.from_wav(seg['audio_path'])
        pos_ms = int(seg['start'] * 1000)
        final_audio = final_audio.overlay(hindi_audio, position=pos_ms)

    if not has_video:
        # Audio-only input — save dubbed audio as WAV alongside the requested output path
        audio_out = output_path.replace('.mp4', '_dubbed.wav')
        final_audio.export(audio_out, format='wav')
        return audio_out

    mixed_audio_path = output_path.replace('.mp4', '_mixed_audio.wav')
    final_audio.export(mixed_audio_path, format='wav')

    video_stream = ffmpeg.input(original_video).video
    audio_stream = ffmpeg.input(mixed_audio_path).audio

    if subtitle_path and os.path.isfile(subtitle_path):
        out = ffmpeg.output(
            video_stream, audio_stream, output_path,
            vf=f"subtitles={subtitle_path}",
            acodec='aac', vcodec='libx264', shortest=None
        )
    else:
        out = ffmpeg.output(
            video_stream, audio_stream, output_path,
            acodec='aac', vcodec='copy', shortest=None
        )

    out.run(overwrite_output=True, quiet=True)
    os.remove(mixed_audio_path)
    return output_path


def generate_srt(segments, output_path):
    """Generate SRT subtitle file from dubbed segments."""
    def fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['start'] + 3.0)}\n")
            f.write(f"{seg.get('text', '')}\n\n")
    return output_path
