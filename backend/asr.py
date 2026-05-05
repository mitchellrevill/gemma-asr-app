"""
asr.py — Gemma 4 E2B-it ASR & structured report extraction.

API contract (from HuggingFace model card google/gemma-4-E2B-it):
  - Model class:   AutoModelForMultimodalLM
  - Processor:     AutoProcessor
  - Audio input:   {"type": "audio", "audio": "<file_path_or_url>"}
  - Chat template: processor.apply_chat_template(messages, tokenize=True,
                     return_dict=True, return_tensors="pt",
                     add_generation_prompt=True)
  - Generate:      model.generate(**inputs, max_new_tokens=512)
  - Decode:        processor.decode(outputs[0][input_len:], skip_special_tokens=False)
  - Parse:         processor.parse_response(response)
"""

import logging
import re
import time
from typing import List

import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_ID = "google/gemma-4-E2B-it"

# ── ASR transcription prompt ────────────────────────────────────────────────
ASR_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three.\n"
    "* Do not add commentary, headers, or explanations — raw transcript only."
)

# ── Tool schema for structured report extraction ─────────────────────────────
MAX_EXTRACT_RETRIES = 3

REPORT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_police_report",
        "description": (
            "Submit a structured police situation report extracted from an "
            "officer's audio transcription. Use null for unknown fields and "
            "empty arrays for missing lists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "incident_number":  {"type": "string", "description": "Incident reference number, or null"},
                "report_date":      {"type": "string", "description": "Report date ISO YYYY-MM-DD, or null"},
                "status":           {"type": "string", "enum": ["open", "closed", "pending"]},
                "classification":   {"type": "string", "enum": ["UNCLASSIFIED", "RESTRICTED", "CONFIDENTIAL"]},
                "incident_type":    {"type": "string", "description": "Category of incident"},
                "incident_date":    {"type": "string", "description": "Incident date ISO YYYY-MM-DD, or null"},
                "incident_time":    {"type": "string", "description": "Incident time HH:MM, or null"},
                "priority":         {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "location":         {"type": "string", "description": "Location description, or null"},
                "grid_ref":         {"type": "string", "description": "Grid reference, or null"},
                "officer_name":     {"type": "string", "description": "Reporting officer full name, or null"},
                "badge_number":     {"type": "string", "description": "Officer badge number, or null"},
                "rank":             {"type": "string", "description": "Officer rank, or null"},
                "unit":             {"type": "string", "description": "Officer unit, or null"},
                "parties": {
                    "type": "array",
                    "description": "Suspects, victims, and witnesses",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role":        {"type": "string", "enum": ["suspect", "victim", "witness"]},
                            "name":        {"type": "string"},
                            "dob":         {"type": "string"},
                            "address":     {"type": "string"},
                            "description": {"type": "string"},
                            "notes":       {"type": "string"},
                        },
                        "required": ["role", "name"],
                    },
                },
                "vehicles": {
                    "type": "array",
                    "description": "Vehicles involved",
                    "items": {
                        "type": "object",
                        "properties": {
                            "registration": {"type": "string"},
                            "make_model":   {"type": "string"},
                            "colour":       {"type": "string"},
                            "owner":        {"type": "string"},
                            "notes":        {"type": "string"},
                        },
                    },
                },
                "narrative":          {"type": "string", "description": "Full cleaned-up narrative"},
                "key_points":         {"type": "array", "items": {"type": "string"}, "description": "Most important facts"},
                "actions_taken":      {"type": "array", "items": {"type": "string"}, "description": "Actions the officer has taken"},
                "follow_up_required": {"type": "boolean"},
                "follow_up_assignee": {"type": "string", "description": "Follow-up assignee, or null"},
                "follow_up_notes":    {"type": "string", "description": "Follow-up notes, or null"},
            },
            "required": [
                "incident_type", "narrative", "key_points", "actions_taken",
                "status", "classification", "priority", "follow_up_required",
            ],
        },
    },
}


class GemmaASR:
    """Wraps Gemma 4 E2B-it for ASR transcription + police report extraction."""

    # Quantization config — order of precedence, highest first:
    #   1. Env var GEMMA_QUANTIZE=1 / =0   (runtime override, e.g. for demos)
    #   2. Class attribute GEMMA_QUANTIZE  (file-level default; True / False / None)
    #   3. Auto: quantize if VRAM < _LOW_VRAM_THRESHOLD_GB
    GEMMA_QUANTIZE = 0
    _LOW_VRAM_THRESHOLD_GB = 8

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None

    def _should_quantize(self, vram_gb: float | None) -> bool:
        env = os.environ.get("GEMMA_QUANTIZE")
        if env == "1":
            return True
        if env == "0":
            return False
        if self.GEMMA_QUANTIZE is not None:
            return bool(self.GEMMA_QUANTIZE)
        return vram_gb is not None and vram_gb < self._LOW_VRAM_THRESHOLD_GB

    def load(self):
        """Lazy-load the model and processor."""
        if self._model is not None:
            return
        use_cuda = self.device == "cuda" and torch.cuda.is_available()

        vram_gb = None
        if use_cuda:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            logger.info(f"GPU detected: {props.name} ({vram_gb:.1f} GB VRAM)")

        quantize = self._should_quantize(vram_gb)
        device_map = "cuda" if use_cuda else "cpu"

        model_kwargs = {
            "device_map": device_map,
            "attn_implementation": "sdpa",
        }

        if quantize:
            # Quantize the language transformer to 4-bit NF4, but keep the
            # audio/vision encoders and the output head in bf16 — Gemma 4's
            # audio tower calls torch.finfo() on its own weight dtype during
            # forward, which crashes if the weights are 4-bit.
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=[
                    "audio_tower",
                    "vision_tower",
                    "embed_audio",
                    "embed_vision",
                    "lm_head",
                ],
            )
            # Keep skipped modules in bf16 (not the default fp32).
            model_kwargs["dtype"] = torch.bfloat16
            logger.info(
                f"Loading {MODEL_ID} on {device_map} with 4-bit NF4 quantization "
                f"(low-VRAM mode, audio/vision towers kept in bf16) — "
                f"this may take several minutes on first run…"
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16 if use_cuda else torch.float32
            logger.info(
                f"Loading {MODEL_ID} on {device_map} ({model_kwargs['dtype']}) — "
                f"this may take several minutes on first run…"
            )

        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **model_kwargs)
        self._model.eval()
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    @property
    def model(self):
        self.load()
        return self._model

    @property
    def processor(self):
        self.load()
        return self._processor

    def _normalize_parse_response(self, response):
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("text", "transcription", "transcript"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
            string_values = [v for v in response.values() if isinstance(v, str)]
            if len(string_values) == 1:
                return string_values[0]
            if string_values:
                return " ".join(string_values)
            return " ".join(str(v) for v in response.values())
        return str(response)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _extract_tool_calls_regex(self, text: str) -> list:
        """Regex fallback: parse <|tool_call>...</tool_call|> tokens from raw output."""
        def _cast(v: str):
            v = v.strip()
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            try:
                return int(v)
            except ValueError:
                pass
            try:
                return float(v)
            except ValueError:
                pass
            return v.strip("'\"")

        # The model usually closes with `<tool_call|>` but sometimes (especially
        # under quantization) emits `<turn|>` / `<end_of_turn|>` / `<eos>` or
        # nothing at all. Accept any of these as the call terminator.
        end_marker = r"(?:<tool_call\|>|<turn\|>|<end_of_turn\|>|<eos>|\Z)"
        results = []
        for name, args in re.findall(
            r"<\|tool_call>call:(\w+)\{(.*?)\}\s*" + end_marker,
            text,
            re.DOTALL,
        ):
            arguments = {
                k: _cast(v1 or v2)
                for k, v1, v2 in re.findall(
                    r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args
                )
            }
            results.append({"name": name, "arguments": arguments})
        return results

    # ── Core inference ────────────────────────────────────────────────────

    def transcribe_chunk(self, audio_path: str) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio_path: Absolute path to a WAV file (≤28 seconds).

        Returns:
            Transcription string.
        """
        # The model accepts a file path directly in the audio field
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": ASR_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        return self._normalize_parse_response(
            self.processor.parse_response(response)
        ).strip()

    def transcribe_chunks(
        self, chunk_paths: List[str], on_progress=None
    ) -> str:
        """
        Transcribe multiple chunks and concatenate the results.

        Args:
            chunk_paths: Ordered list of WAV file paths.
            on_progress: Optional callback(chunk_idx, total) for progress updates.

        Returns:
            Full concatenated transcription.
        """
        segments = []
        total = len(chunk_paths)
        for i, path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{total}: {path}")
            text = self.transcribe_chunk(path)
            segments.append(text)
            if on_progress:
                on_progress(i + 1, total)
        return " ".join(segments)

    def extract_report(self, transcription: str) -> dict:
        """
        Use Gemma 4 function calling to extract a structured police report.
        Retries up to MAX_EXTRACT_RETRIES times before falling back.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a police report analyst. "
                    "Extract all available information from the transcription "
                    "and call submit_police_report with the structured data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract a structured police report from this transcription "
                    "and call submit_police_report:\n\n" + transcription
                ),
            },
        ]

        for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
            try:
                logger.info(f"extract_report attempt {attempt}: rendering prompt with tools…")
                t_render = time.time()
                inputs = self.processor.apply_chat_template(
                    messages,
                    tools=[REPORT_TOOL_SCHEMA],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(self.model.device)
                input_len = inputs["input_ids"].shape[-1]
                logger.info(
                    f"extract_report attempt {attempt}: prompt rendered in "
                    f"{time.time() - t_render:.2f}s, input_len={input_len} tokens"
                )

                t_gen = time.time()
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                    )
                logger.info(
                    f"extract_report attempt {attempt}: generate() done in "
                    f"{time.time() - t_gen:.2f}s, generated "
                    f"{outputs.shape[-1] - input_len} tokens"
                )

                raw = self.processor.decode(
                    outputs[0][input_len:], skip_special_tokens=False
                )
                logger.info(
                    f"=== RAW MODEL OUTPUT (attempt {attempt}, {len(raw)} chars) ===\n"
                    f"{raw}\n=== END RAW ==="
                )
                parsed = self.processor.parse_response(raw)
                logger.info(f"=== PARSED (attempt {attempt}) ===\n{parsed!r}\n=== END PARSED ===")

                # parse_response returns a dict with tool_calls on success
                tool_calls = None
                if isinstance(parsed, dict):
                    tool_calls = parsed.get("tool_calls")

                # Fallback: regex extraction from raw token stream
                if not tool_calls:
                    regex_calls = self._extract_tool_calls_regex(raw)
                    if regex_calls:
                        tool_calls = [{"function": c} for c in regex_calls]

                if tool_calls:
                    for call in tool_calls:
                        fn = call.get("function", call)
                        if fn.get("name") == "submit_police_report":
                            return fn.get("arguments", {})
                    logger.warning(
                        f"Attempt {attempt}: unexpected function(s): "
                        + str([c.get("function", c).get("name") for c in tool_calls])
                    )
                else:
                    logger.warning(f"Attempt {attempt}: no tool call found in response")

            except Exception as exc:
                logger.warning(f"extract_report attempt {attempt} raised: {exc}")

            if attempt < MAX_EXTRACT_RETRIES:
                logger.info(f"Retrying extraction ({attempt + 1}/{MAX_EXTRACT_RETRIES})…")

        logger.error("All extraction attempts failed; returning fallback report")
        return {
            "narrative": transcription,
            "key_points": ["(Structured extraction failed — raw transcription preserved)"],
            "actions_taken": [],
            "follow_up_required": False,
            "status": "open",
            "classification": "UNCLASSIFIED",
            "priority": "MEDIUM",
        }


# Module-level singleton — shared across all FastAPI requests
_asr_instance: GemmaASR | None = None


def get_asr() -> GemmaASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = GemmaASR()
    return _asr_instance
