"""
audio_utils.py — Audio loading and chunking utilities for the Gemma ASR pipeline.

Chunks audio into 28-second segments to fit within the model's context window.
"""

import os
import math
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

CHUNK_DURATION_S = 28  # seconds — fits Gemma 4 E2B-it audio context window
TARGET_SAMPLE_RATE = 16_000  # 16 kHz — standard for ASR models


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resampling to TARGET_SAMPLE_RATE (16 kHz mono).

    Returns:
        (waveform, sample_rate) — waveform is a 1-D float32 numpy array.
    """
    logger.info(f"Loading audio: {file_path}")
    waveform, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    duration = len(waveform) / sr
    logger.info(f"Loaded {duration:.1f}s of audio at {sr} Hz")
    return waveform, sr


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_duration_s: int = CHUNK_DURATION_S,
) -> List[np.ndarray]:
    """
    Split a waveform into fixed-length chunks.

    Args:
        waveform:          1-D float32 numpy array.
        sample_rate:       Sample rate of the waveform.
        chunk_duration_s:  Maximum duration of each chunk in seconds.

    Returns:
        List of waveform chunks (numpy arrays).
    """
    chunk_size = chunk_duration_s * sample_rate
    n_chunks = math.ceil(len(waveform) / chunk_size)
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = waveform[start:end]
        chunks.append(chunk)
    logger.info(f"Split into {n_chunks} chunks of ≤{chunk_duration_s}s each")
    return chunks


def save_chunks_to_temp(
    chunks: List[np.ndarray],
    sample_rate: int,
    temp_dir: str | None = None,
) -> List[str]:
    """
    Save each audio chunk to a temporary WAV file.

    Args:
        chunks:       List of waveform chunks.
        sample_rate:  Sample rate to write.
        temp_dir:     Optional directory for temp files (uses system temp if None).

    Returns:
        List of absolute file paths to the written WAV files.
    """
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)

    paths = []
    for i, chunk in enumerate(chunks):
        fd, path = tempfile.mkstemp(
            suffix=f"_chunk_{i:03d}.wav",
            dir=temp_dir,
        )
        os.close(fd)
        sf.write(path, chunk, sample_rate, subtype="PCM_16")
        paths.append(path)
        logger.debug(f"Saved chunk {i} → {path}")
    return paths


def cleanup_temp_files(paths: List[str]) -> None:
    """Remove temporary chunk files."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def get_audio_duration(file_path: str) -> float:
    """Return the duration of an audio file in seconds without loading it fully."""
    return librosa.get_duration(path=file_path)
