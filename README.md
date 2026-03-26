# Outreach Report Generator

An end-to-end pipeline that converts multi-speaker agricultural outreach audio recordings into structured, translated, insight-rich reports. Designed for large-scale rural interactions - where farmers are introduced to an AI agri-chatbot and openly discuss their challenges, practices, and feedback - the system automates everything from raw audio to polished outputs (PDF, Excel, Word) for analysis, documentation, and decision-making.

---

## Overview

The pipeline takes a directory of audio files in a supported Indic language, diarizes speakers, transcribes speech using a state-of-the-art ASR model, translates to English, extracts structured insights, and assembles a PDF report.

```
Audio files (wav / mp3 / m4a / flac)
    └─► Combine & normalize
        └─► Speaker diarization       (pyannote.audio)
            └─► ASR transcription     (IndicConformer)
                └─► Translation       (Sarvam AI)
                    └─► Extraction    (LLM-based: insights, participants, terminology, metadata, conclusion)
                        └─► PDF Report (ReportLab)
```

---

## Features

- **Multi-speaker diarization** - identifies and separates speakers using `pyannote.audio`
- **Indic ASR** - transcribes speech in 10 Indian languages via `IndicConformer`
- **Neural machine translation** - translates Indic-language transcripts to English using Sarvam AI
- **LLM-based extraction** - concurrently extracts farmer questions, challenges, participant info, domain terminology, meeting metadata, and a narrative conclusion
- **PDF report generation** - assembles all extracted content into a formatted report using ReportLab
- **Resumable pipeline** - skip any completed stage (`--skip_combine`, `--skip_asr`, `--skip_translation`) to avoid re-running expensive steps
- **GPU-accelerated** - automatically uses CUDA when available, with explicit memory management between stages

---

## Supported Languages

| Code | Language   | IndicTrans2 Tag | ASR Tag |
|------|------------|-----------------|---------|
| `pa` | Punjabi    | `pan_Guru`      | `pa`    |
| `hi` | Hindi      | `hin_Deva`      | `hi`    |
| `ta` | Tamil      | `tam_Taml`      | `ta`    |
| `te` | Telugu     | `tel_Telu`      | `te`    |
| `mr` | Marathi    | `mar_Deva`      | `mr`    |
| `kn` | Kannada    | `kan_Knda`      | `kn`    |
| `gu` | Gujarati   | `guj_Gujr`      | `gu`    |
| `bn` | Bengali    | `ben_Beng`      | `bn`    |
| `or` | Odia       | `ory_Orya`      | `or`    |
| `ml` | Malayalam  | `mal_Mlym`      | `ml`    |

---

## Project Structure

```
outreach-report-generator/
├── main.py                        # Pipeline entry point & CLI
├── pipeline/
│   ├── ingestion/
│   │   └── audio_utils.py         # Audio combining & normalization
│   ├── diarization/
│   │   └── pyannote_diarizer.py   # Speaker diarization
│   ├── asr/
│   │   └── indic_conformer.py     # IndicConformer ASR model
│   ├── transcript/
│   │   └── builder.py             # Builds & serializes structured transcripts
│   ├── translation/
│   │   └── sarvam_translate.py    # Sarvam AI translation (Indic → English)
│   ├── extraction/
│   │   ├── base_llm.py            # Shared LLM model loader
│   │   ├── insights.py            # Farmer questions & challenges extractor
│   │   ├── narration.py           # Narrative text generator
│   │   ├── conclusion.py          # Conclusion generator
│   │   ├── metadata.py            # Meeting metadata extractor
│   │   ├── participants.py        # Participant extractor
│   │   └── terminology.py         # Domain terminology extractor
│   └── report/
│       └── assembler.py           # Assembles & saves final report (JSON + PDF)
├── config/                        # Configuration files
├── scripts/                       # Utility scripts
├── tests/                         # Test suite
├── assets/
│   └── fonts/                     # Fonts for PDF generation
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Project metadata
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended; CPU fallback is supported but slow)
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/scriptforge-ds/outreach-report-generator.git
cd outreach-report-generator

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys (see Configuration section)
```

---

## Configuration

Copy `.env.example` to `.env` and populate the required keys.

The pipeline uses the following external services that require credentials:

| Service | Purpose | Where to obtain |
|---------|---------|-----------------|
| **pyannote.audio** | Speaker diarization model (gated on HuggingFace) | [hf.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) - accept terms & generate a token |
| **Sarvam AI** | Indic-language translation | [sarvam.ai](https://www.sarvam.ai) |
| **HuggingFace Hub** | Model downloads (IndicConformer, IndicTrans2) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Usage

```bash
python main.py --input_dir <path_to_audio> --language <lang_code> [options]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | ✅ | — | Directory containing audio files (wav, mp3, m4a, flac) |
| `--language` | ✅ | — | Source language code (see Supported Languages table) |
| `--output_dir` | — | `./outputs` | Directory for all outputs |
| `--skip_combine` | — | `false` | Reuse existing `combined.wav` |
| `--skip_asr` | — | `false` | Reuse existing `transcript_raw.json` |
| `--skip_translation` | — | `false` | Reuse existing `transcript_translated.json` |
| `--no_pdf` | — | `false` | Skip PDF generation; save JSON outputs only |

### Examples

```bash
# Basic run - Punjabi audio
python main.py --input_dir ./audio --language pa

# Custom output directory - Hindi audio
python main.py --input_dir ./audio --language hi --output_dir ./outputs/meeting_001

# Skip ASR (reuse a previously generated transcript)
python main.py --input_dir ./audio --language pa --skip_asr

# Skip both ASR and translation
python main.py --input_dir ./audio --language pa --skip_asr --skip_translation

# Skip PDF generation
python main.py --input_dir ./audio --language hi --no_pdf
```

---

## Pipeline Stages

### Stage 1 - Audio Ingestion
All audio files in `--input_dir` are combined into a single normalized WAV file (`combined.wav`) for downstream processing.

### Stage 2 - Speaker Diarization
`pyannote.audio` identifies speaker turns, producing timestamped speaker segments.

### Stage 3 - ASR Transcription
`IndicConformer` transcribes each speaker segment in the source Indic language, producing a structured raw transcript (`transcript_raw.json`).

### Stage 4 - Translation
Sarvam AI translates each transcript segment from the source Indic language to English, producing `transcript_translated.json`.

### Stage 5 - Extraction (Concurrent)
An LLM processes the translated transcript to concurrently extract:
- **Terminology** - domain-specific vocabulary from the session
- **Narration** - a flowing narrative summary of the session (up to 20,000 characters)
- **Insights** - farmer questions and challenges raised during the session
- **Participants** - count and roles of attendees
- **Metadata** - meeting date, location, topic, and other structured fields
- **Conclusion** - a synthesized conclusion drawn from participants, challenges, questions, and narration

> Model weights are loaded once via `BaseLLM` and shared across all extractors to minimize GPU memory usage.

### Stage 6 - Report Assembly
All extracted data is assembled into a structured JSON and, optionally, a formatted PDF report (`outreach_report.pdf`) via ReportLab.

---

## Outputs

All outputs are saved to `--output_dir` (default: `./outputs`):

| File | Description |
|------|-------------|
| `combined.wav` | Merged and normalized input audio |
| `transcript_raw.json` | Speaker-diarized ASR transcript in source language |
| `transcript_translated.json` | English-translated transcript |
| `outreach_report.pdf` | Final assembled PDF report (unless `--no_pdf`) |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchaudio` | Deep learning backend |
| `transformers` | IndicConformer ASR & IndicTrans2 translation models |
| `pyannote-audio` | Speaker diarization |
| `reportlab` | PDF generation |
| `accelerate` | HuggingFace model acceleration |
| `python-dotenv` | Environment variable management |

See `requirements.txt` for the full pinned dependency list.

---

Here’s a clean, brief **Future Improvements / Limitations** section you can append:

---

## Future Improvements

While the pipeline delivers strong end-to-end automation, several areas are being actively improved:

* **ASR robustness** – Current performance drops for low-resource dialects, long/noisy field recordings, and domain-specific agricultural vocabulary
* **Translation fidelity** – Some loss of nuance and contextual meaning when converting from Indic languages to English
* **LLM extraction consistency** – Variability across runs in structured outputs (insights, participants, metadata)
* **Hallucinations & accuracy** – Occasional generation of unsupported facts and inconsistent extraction of key details (e.g., farmer names, event time)
* **Insight quality** – Irrelevant or noisy outputs in question/challenge extraction and weaker domain terminology identification

> Ongoing work focuses on domain adaptation, improved prompting and validation strategies, and model-level enhancements to increase robustness, factual accuracy, and consistency across pipeline stages.

---

## License

See [LICENSE](LICENSE) for details.