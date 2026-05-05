# Police Situation Report

A small local web app that takes an officer's verbal report  recorded in the
browser or uploaded as a file  and turns it into a structured incident report
you can review, print, or export as JSON.

Everything runs on your machine. The model is Gemma 4 E2B-it (audio + text)
loaded through Hugging Face transformers; nothing is sent to a third-party
service.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with at least 6 GB of VRAM and a reasonably recent driver.
  Smaller GPUs (4 GB) work too — the model loads in 4-bit NF4 automatically.
  CPU inference *works* but is impractical (see [Notes](#notes)).
- ~6 GB of free disk for the model weights, cached on first run

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

The default `requirements.txt` pulls the CUDA 12.8 build of PyTorch. If you
need a different CUDA version, edit the `--extra-index-url` line and the
`torch==…+cu128` pin to match (e.g. `cu124`, `cu126`).

Sanity check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Running

```
run.bat
```

…then open <http://localhost:8000>.

First start downloads the model (~6 GB). Subsequent starts load from the local
HF cache in about 15 seconds.

## How it works

1. The browser sends the audio file to `POST /api/transcribe`.
2. `audio_utils.py` resamples it to 16 kHz mono and splits it into segments
   that fit the model's audio context window.
3. `asr.py` runs the model over each segment with a transcription prompt and
   stitches the text back together.
4. The same model is then called a second time with a function/tool schema
   describing every field of a police report. It returns a single tool call
   whose arguments *are* the report — incident type, parties, vehicles,
   narrative, key points, follow-up, etc.
5. The JSON is rendered into a printable HTML template in the right pane of
   the UI.

## Layout

```
backend/
  main.py          FastAPI app, request handling, lifecycle
  asr.py           Model loading, transcription, structured extraction
  audio_utils.py   Loading, resampling, chunking
  requirements.txt
frontend/
  index.html       Single-page UI: recorder, uploader, report pane
templates/
  police_report_template.html   Printable report layout
run.bat
```

## Notes

- CPU inference is memory-bandwidth bound and runs at roughly 2 tokens per
  second on a typical desktop. A medium-length report can take five minutes
  or more per request. Get a GPU.
- On GPUs under 8 GB VRAM, the model auto-loads in 4-bit NF4 quantization
  (via `bitsandbytes`). This shrinks the weights to ~2.5 GB at the cost of
  a small accuracy hit. Force on/off with `GEMMA_QUANTIZE=1` or `=0`.
- The default attention backend is PyTorch's built-in `sdpa`. Installing
  `flash-attn` and switching `attn_implementation` to `"flash_attention_2"`
  in `asr.py` will pick up another ~30% on the long prompt-extraction step,
  but it needs the CUDA toolkit on disk and isn't worth the trouble for
  most setups.
- Hugging Face will let you download the model unauthenticated, but
  setting `HF_TOKEN` in your environment gets you faster downloads and
  higher rate limits.
