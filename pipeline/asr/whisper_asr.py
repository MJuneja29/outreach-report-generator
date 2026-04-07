import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_asr_model(device: str):
    print("Loading Whisper ASR model...")
    model_id = "openai/whisper-large-v3"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        # Default internal chunking for inputs > 30s
        chunk_length_s=30,
        stride_length_s=5,
    )
    return pipe


# =============================================================================
# INFERENCE
# =============================================================================

def transcribe_chunk(
    pipe,
    chunk: torch.Tensor,
    language: str,
    device: str,
) -> str:
    """
    Run Whisper ASR on a single audio chunk.
    """
    try:
        # Convert torch tensor [1, N] to flat numpy array for pipeline
        audio_np = chunk.squeeze(0).cpu().numpy()
        
        # Add basic prompt handling placeholder (we will use this later)
        generate_kwargs = {"language": language, "task": "transcribe"}

        result = pipe(
            audio_np,
            generate_kwargs=generate_kwargs
        )
        return result["text"].strip()
    except Exception as e:
        print(f"    ASR failed: {e}")
        return ""
