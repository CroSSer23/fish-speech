import html
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

OUTPUT_DIR = Path("outputs")


def _auto_save(audio: tuple, filepath: Path) -> Path | None:
    try:
        sample_rate, data = audio
        filepath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(filepath), data, sample_rate)
        return filepath
    except Exception as e:
        print(f"[auto-save] failed: {e}")
        return None


def _make_request(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
) -> ServeTTSRequest:
    references = []
    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)

    return ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )


def _run_inference(req: ServeTTSRequest, engine):
    """Run inference and return (sample_rate, data) or raise."""
    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio
            case "error":
                raise RuntimeError(result.error)
    return None


def inference_wrapper(
    text,
    mode,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """Wrapper for the inference function. Used in the Gradio interface."""

    common = dict(
        reference_id=reference_id,
        reference_audio=reference_audio,
        reference_text=reference_text,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=seed,
        use_memory_cache=use_memory_cache,
    )

    if mode == "Line-by-line":
        return _inference_linewise(text, engine, **common)
    else:
        return _inference_normal(text, engine, **common)


# ── Normal mode ───────────────────────────────────────────────────────────────

def _inference_normal(text, engine, **common):
    try:
        req = _make_request(text=text, **common)
        audio = _run_inference(req, engine)
    except RuntimeError as e:
        return None, build_html_error_message(i18n(e))

    if audio is None:
        return None, i18n("No audio generated")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = _auto_save(audio, OUTPUT_DIR / f"{timestamp}.wav")
    if saved:
        print(f"[auto-save] Saved: {saved}")

    sample_rate, data = audio
    data_i16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    return (sample_rate, data_i16), None


# ── Line-by-line mode ─────────────────────────────────────────────────────────

def _inference_linewise(text, engine, **common):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, build_html_error_message(Exception("No lines to process"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    last_audio = None
    errors = []

    for idx, line in enumerate(lines, start=1):
        print(f"[line-by-line] {idx}/{len(lines)}: {line[:60]}...")
        try:
            req = _make_request(text=line, **common)
            audio = _run_inference(req, engine)
        except RuntimeError as e:
            errors.append(f"Line {idx}: {e}")
            continue

        if audio is None:
            errors.append(f"Line {idx}: No audio generated")
            continue

        saved = _auto_save(audio, batch_dir / f"{idx}.wav")
        if saved:
            print(f"[auto-save] Saved: {saved}")

        last_audio = audio

    if last_audio is None:
        msg = "No audio generated. " + "; ".join(errors)
        return None, build_html_error_message(Exception(msg))

    error_html = build_html_error_message(Exception("; ".join(errors))) if errors else None

    sample_rate, data = last_audio
    data_i16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    return (sample_rate, data_i16), error_html


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    with open(reference_audio, "rb") as f:
        audio_bytes = f.read()
    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:
    error = error if isinstance(error, Exception) else Exception("Unknown error")
    return f"""
    <div style="color: red; font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    return partial(inference_wrapper, engine=engine)
