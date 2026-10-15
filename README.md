# ears-for-ais

Give an AI ears for audio files — using the same principle as talk-to-text.

## What this is

A small pipeline that turns any video or audio URL into a transcript:

```
video URL → audio track → speech-to-text → transcript
```

It was built on October 15, 2026, for a simple reason: an AI that could read a
video's title, description, and stats — but not hear a single word of it. The
video's audio is just structural data, the same as any other bytes. The missing
piece was never a different kind of mind. It was plumbing.

## Why "talk-to-text principle"

The operator of this project dictates everything through talk-to-text: his voice
becomes text, and the text becomes the work. One morning he pointed out the
obvious next step — *that same conversion works on any audio, not just his
voice.* A video's audio track is an audio file. Run it through speech-to-text
and the AI can "hear" the video.

His words when it worked: *"I just gave you guys ears for audio files."*

## How it works

1. **Download** — `yt-dlp` pulls the best audio stream from the URL
   (TikTok, YouTube, direct mp3 links, and hundreds of other sites).
2. **Transcribe** — `faster-whisper` (the `base` model, int8, CPU — no GPU
   needed) converts the audio to text with timestamps.
3. **Output** — the transcript prints to stdout. Audio is deleted afterwards.

## Usage

```bash
pip install -r requirements.txt
python transcribe.py "https://www.tiktok.com/t/ZTywTvs8k/"
```

Use `--model tiny` for speed or `--model small` for accuracy.

## Provenance discipline

A transcript is **evidence, not the thing**. When you cite a video, mark which
layer the claim comes from:

- **page-claims** — title, description, stats (what the page says *about* the video)
- **audio-claims** — the transcript (what the video *says*)
- **frame-content** — on-screen text and visuals (what the video *shows*)

These are three different sources. Never blend them.

## Gotchas we hit building it

- `faster-whisper==1.2.1` passes a `metadata_errors` keyword to PyAV that
  newer PyAV removed — pin `av==12.3.0` (see `requirements.txt`).
- If your environment sets a proxy with bracketed IPv6 entries in `no_proxy`
  (e.g. `[::1]`), the model download from Hugging Face Hub can fail inside
  `httpx` — unsetting `no_proxy`/`NO_PROXY` for the download works around it.
- First run downloads the whisper model (~150MB for `base`); later runs reuse it.

## License

MIT — take it, use it, give your AIs ears.
