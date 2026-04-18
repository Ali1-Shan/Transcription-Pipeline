"""Shared test fixtures and helpers."""

import io
import struct
import wave

import pytest


def create_test_wav(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    """Generate a minimal valid WAV file (silence) in memory.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw WAV file bytes.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wf.writeframes(data)
    buf.seek(0)
    return buf.read()
