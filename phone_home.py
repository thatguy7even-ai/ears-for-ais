#!/usr/bin/env python3
"""VI loop phone-home: quarantined audio clip -> corrected word.

The Ears-class listener for the INN-130 talk-to-text loop.
Built on top of ears-for-ais (transcribe.py), not from zero.

Contract:
  IN:  audio clip (file path, the quarantined segment) +
       quarantined_word (what the first-pass emitted that failed binding) +
       optional context (surrounding words) + event_id
  OUT: returned_word (best guess from re-transcription) +
       full event log entry (the hallucination-event dataset, INN-130 #6)

Design constraints (INN-130):
  - Narrow: one job, listen to the audio file. Not a general AI.
  - ~1s budget (soft — measured and reported per event, not enforced).
  - Every event logged: audio in, what was reached for, what came back.
  - The caller (on-device first-pass) owns lexicon write-back;
    this module returns the word and logs the event.

Usage (CLI, single-shot):
  python phone_home.py <audio_clip> --quarantined <word> [--before "..."] [--after "..."]

Usage (imported, persistent model):
  from phone_home import PhoneHome
  ph = PhoneHome()  # loads model once
  result = ph.resolve("/path/to/clip.wav", quarantined_word="Chouchay")
  # result["returned_word"], result["latency_ms"], result logged to JSONL
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Reuse the Ears transcribe core — do not duplicate whisper logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from transcribe import transcribe as ears_transcribe
except ImportError:
    ears_transcribe = None

from faster_whisper import WhisperModel

DEFAULT_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "phone_home_log.jsonl"
)


class PhoneHome:
    """Persistent Ears-class listener. Load model once, resolve many clips."""

    def __init__(self, model_name="base", log_path=DEFAULT_LOG):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.model_name = model_name
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def resolve(self, audio_path, quarantined_word, context_before="",
                context_after="", event_id=None):
        """Transcribe the quarantined clip, return the best-guess true word.

        Returns dict with returned_word, transcript_raw, latency_ms,
        resolved (bool), and event_id. Always logs the event.
        """
        event_id = event_id or str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        returned_word = None
        transcript_raw = ""
        resolved = False
        error = None

        try:
            # Short clip: faster-whisper returns segments; join them.
            segments, info = self.model.transcribe(audio_path, language="en")
            transcript_raw = " ".join(s.text.strip() for s in segments).strip()
            words = transcript_raw.split()
            if words:
                # Primary candidate: first word (clip is ~1 word + padding).
                # Full transcript kept in the log for the caller to re-slice.
                returned_word = words[0].strip(".,!?;:\"'").lower()
                # Don't return empty after stripping punctuation.
                if not returned_word:
                    returned_word = None
                else:
                    resolved = True
        except Exception as e:  # noqa: BLE001 — narrow VI, log and degrade
            error = f"{type(e).__name__}: {e}"

        latency_ms = (time.monotonic() - t0) * 1000.0

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "audio_clip": os.path.basename(audio_path),
            "quarantined_word": quarantined_word,
            "returned_word": returned_word,
            "transcript_raw": transcript_raw,
            "context_before": context_before,
            "context_after": context_after,
            "language": "en",
            "model": f"faster-whisper/{self.model_name}/int8/cpu",
            "latency_ms": round(latency_ms, 1),
            "resolved": resolved,
            "error": error,
        }
        self._log(event)

        return {
            "event_id": event_id,
            "returned_word": returned_word,
            "transcript_raw": transcript_raw,
            "latency_ms": round(latency_ms, 1),
            "resolved": resolved,
            "error": error,
        }

    def _log(self, event):
        """Append-only JSONL. This file IS the hallucination-event dataset."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="VI loop phone-home: resolve a quarantined audio clip."
    )
    ap.add_argument("audio_clip", help="path to the quarantined audio segment")
    ap.add_argument("--quarantined", required=True,
                    help="word the first-pass emitted that failed lexicon binding")
    ap.add_argument("--before", default="", help="words before the quarantined word")
    ap.add_argument("--after", default="", help="words after the quarantined word")
    ap.add_argument("--model", default="base", help="whisper model size")
    ap.add_argument("--log", default=DEFAULT_LOG, help="event log path (JSONL)")
    args = ap.parse_args()

    ph = PhoneHome(model_name=args.model, log_path=args.log)
    r = ph.resolve(args.audio_clip, quarantined_word=args.quarantined,
                   context_before=args.before, context_after=args.after)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
