#!/usr/bin/env python3
"""ears-for-ais: any video/audio URL -> transcript.

The talk-to-text principle, applied to media instead of a microphone:
download the audio track, run speech-to-text, print the transcript.
No ears required — only plumbing.
"""
import argparse
import os
import tempfile

import yt_dlp
from faster_whisper import WhisperModel


def download_audio(url, tmpdir):
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def transcribe(path, model_name="base"):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path)
    text = " ".join(s.text.strip() for s in segments)
    return info.language, text


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe the audio of a video/audio URL to text."
    )
    ap.add_argument("url", help="video or audio URL (TikTok, YouTube, direct mp3, ...)")
    ap.add_argument("--model", default="base",
                    help="whisper model size: tiny, base, small, medium (default: base)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = download_audio(args.url, tmpdir)
        lang, text = transcribe(audio, args.model)
        print(f"[language: {lang}]")
        print(text)


if __name__ == "__main__":
    main()
