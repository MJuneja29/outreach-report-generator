"""
Thin wrapper around pyannote/speaker-diarization-community-1.
Returns a list of (start, end, speaker_id) turn tuples.

Requires HF_TOKEN in .env — the pyannote model is gated on HuggingFace.
"""

import os

import soundfile as sf
import torch
from pyannote.audio import Pipeline


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_diarization_pipeline(device: str) -> Pipeline:
    # Safely hardcoded per user request
    token = "enter the token"

    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    pipeline.to(torch.device(device))

    return pipeline


# =============================================================================
# INFERENCE
# =============================================================================

def diarize(pipeline: Pipeline, audio_path: str) -> list[tuple[float, float, str]]:
    """
    Run full-audio diarization and return a sorted list of
    (start_sec, end_sec, speaker_id) tuples.
    """
    print("Running speaker diarization...")
    # Load via soundfile bypassing torchaudio entirely
    wav_np, sr = sf.read(audio_path, dtype="float32")
    if wav_np.ndim == 1:
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
    else:
        wav_tensor = torch.from_numpy(wav_np).transpose(0, 1)
        wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)
        
    in_memory_audio = {"waveform": wav_tensor, "sample_rate": sr}
    diarization = pipeline(in_memory_audio)
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
    ]
    print(f"Found {len(turns)} speaker turns.")
    return turns