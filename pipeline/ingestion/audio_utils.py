"""
Handles discovery, loading, normalization, and combining of raw audio files
into a single 16kHz mono WAV ready for ASR + diarization.
"""

import glob
import os

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment


# =============================================================================
# FIND AND SORT FILES
# =============================================================================

def get_sorted_files(directory: str) -> list[str]:
    """
    Discover all audio files in a directory and return them in pipeline order:
    narration file(s) first, then all others sorted alphabetically.
    """
    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    if not files:
        raise FileNotFoundError(f"No audio files found in {directory}")

    narration_files = [f for f in files if "narration" in os.path.basename(f).lower()]
    other_files     = sorted(
        [f for f in files if "narration" not in os.path.basename(f).lower()],
        key=lambda f: os.path.basename(f).lower(),
    )

    ordered = narration_files + other_files

    print("File order:")
    for i, f in enumerate(ordered):
        print(f"  {i+1}. {os.path.basename(f)}")

    return ordered


# =============================================================================
# LOAD + NORMALIZE
# =============================================================================

def load_and_normalize(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file, convert to mono, and resample to target_sr."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    
    samples = np.array(audio.get_array_of_samples())
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = (samples.astype(np.float32) - 128.0) / 128.0

    wav = torch.from_numpy(samples).unsqueeze(0)
    return wav  # shape: (1, samples)


# =============================================================================
# COMBINE + SAVE
# =============================================================================

def combine_audio(
    directory: str,
    output_path: str,
    target_sr: int = 16000,
) -> str:
    """
    Load all audio files from directory in order, concatenate them,
    and save as a single 16kHz mono PCM WAV.

    Returns the output_path for chaining.
    """
    files = get_sorted_files(directory)

    chunks = []
    for path in files:
        print(f"Loading: {os.path.basename(path)}")
        wav = load_and_normalize(path, target_sr)
        duration = wav.shape[1] / target_sr
        print(f"  → {duration:.2f}s | shape: {wav.shape}")
        chunks.append(wav)

    combined = torch.cat(chunks, dim=1)
    total_duration = combined.shape[1] / target_sr
    print(f"\nCombined duration: {total_duration:.2f}s")

    # Export using PyDub to completely bypass any C level libsndfile errors
    audio_data = (combined.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    seg = AudioSegment(
        audio_data.tobytes(),
        frame_rate=target_sr,
        sample_width=2,
        channels=1
    )
    seg.export(output_path, format="wav")
    print(f"Saved → {output_path}")
    return output_path