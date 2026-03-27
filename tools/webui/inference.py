import html
import threading
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import numpy as np
import soundfile as sf

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

OUTPUT_DIR = Path("outputs")

_cancel_event = threading.Event()


def cancel_generation():
    """Called by the Stop button — signals all running inference loops to abort."""
    _cancel_event.set()

FADE_MS = 80  # fade-in / fade-out duration in milliseconds


def _apply_fades(data: np.ndarray, sample_rate: int, fade_ms: int = FADE_MS) -> np.ndarray:
    """Apply a short linear fade-in and fade-out to a float32 audio array."""
    n_fade = int(sample_rate * fade_ms / 1000)
    if data.ndim == 1:
        length = len(data)
    else:
        length = data.shape[0]
    n_fade = min(n_fade, length // 4)  # never fade more than 25% of the clip
    if n_fade <= 0:
        return data
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    out = data.copy().astype(np.float32)
    if out.ndim == 1:
        out[:n_fade] *= ramp
        out[-n_fade:] *= ramp[::-1]
    else:
        out[:n_fade] *= ramp[:, np.newaxis]
        out[-n_fade:] *= ramp[::-1, np.newaxis]
    return out


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
    """Yields (segment_count, audio_or_None): None until final, then the audio."""
    segment_count = 0
    for result in engine.inference(req):
        match result.code:
            case "segment":
                segment_count += 1
                yield segment_count, None
            case "final":
                yield segment_count, result.audio
                return
            case "error":
                raise RuntimeError(result.error)


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
    _cancel_event.clear()

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
        yield from _inference_linewise(text, engine, **common)
    else:
        yield from _inference_normal(text, engine, **common)


# ── Normal mode ───────────────────────────────────────────────────────────────

def _inference_normal(text, engine, **common):
    chunk_length = common.get("chunk_length", 300)
    estimated = max(1, len(text) // max(chunk_length, 50)) if chunk_length > 0 else 1

    try:
        req = _make_request(text=text, **common)
        audio = None
        segment_count = 0

        for seg_count, result_audio in _run_inference(req, engine):
            if _cancel_event.is_set():
                yield None, None, _cancelled_html(), None
                return
            segment_count = seg_count
            if result_audio is not None:
                audio = result_audio
            else:
                pct = min(90, int(seg_count / estimated * 100))
                yield gr.update(), gr.update(), _progress_html(seg_count, estimated, pct), gr.update()

    except RuntimeError as e:
        yield None, build_html_error_message(i18n(e)), "", None
        return

    if audio is None:
        yield None, i18n("No audio generated"), "", None
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = _auto_save(audio, OUTPUT_DIR / f"{timestamp}.wav")
    if saved:
        print(f"[auto-save] Saved: {saved}")

    sample_rate, data = audio
    data_i16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    yield (sample_rate, data_i16), None, _done_html(segment_count), None


# ── Line-by-line mode ─────────────────────────────────────────────────────────

def _inference_linewise(text, engine, **common):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        yield None, build_html_error_message(Exception("No lines to process")), "", None
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    errors = []
    total_segments = 0

    for idx, line in enumerate(lines, start=1):
        if _cancel_event.is_set():
            break
        print(f"[line-by-line] {idx}/{len(lines)}: {line[:60]}...")
        yield gr.update(), gr.update(), _line_progress_html(idx - 1, len(lines), total_segments), gr.update()

        line_segments = 0
        try:
            req = _make_request(text=line, **common)
            audio = None
            for seg_count, result_audio in _run_inference(req, engine):
                if _cancel_event.is_set():
                    break
                line_segments = seg_count
                if result_audio is not None:
                    audio = result_audio
                else:
                    yield gr.update(), gr.update(), _line_progress_html(idx, len(lines), total_segments + seg_count), gr.update()
        except RuntimeError as e:
            errors.append(f"Line {idx}: {e}")
            continue

        total_segments += line_segments

        if audio is None:
            errors.append(f"Line {idx}: No audio generated")
            continue

        sample_rate, data = audio
        data_faded = _apply_fades(data.astype(np.float32), sample_rate)
        saved = _auto_save((sample_rate, data_faded), batch_dir / f"{idx}.wav")
        if saved:
            print(f"[auto-save] Saved: {saved}")
            saved_paths.append(saved)

    cancelled = _cancel_event.is_set()

    if not saved_paths:
        msg = "Cancelled — no files saved." if cancelled else "No audio generated. " + "; ".join(errors)
        yield None, build_html_error_message(Exception(msg)), "", None
        return

    zip_path = batch_dir.parent / f"{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in saved_paths:
            zf.write(p, p.name)
    print(f"[line-by-line] ZIP saved: {zip_path}")

    error_html = build_html_error_message(Exception("; ".join(errors))) if errors else None
    status_html = _cancelled_html(len(saved_paths)) if cancelled else _done_html(total_segments)
    yield None, error_html, status_html, str(zip_path)


# ── Progress HTML helpers ─────────────────────────────────────────────────────

def _progress_html(done: int, estimated: int, pct: int) -> str:
    return f"""
    <div style="padding:6px 0">
      <div style="font-size:13px;color:#555;margin-bottom:5px">
        🔄 Segment {done} / ~{estimated} estimated
      </div>
      <div style="background:#e0e0e0;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:#4A90D9;height:100%;width:{pct}%;transition:width 0.3s ease;border-radius:6px"></div>
      </div>
    </div>
    """


def _line_progress_html(done: int, total: int, segments: int) -> str:
    pct = int(done / total * 100) if total else 0
    return f"""
    <div style="padding:6px 0">
      <div style="font-size:13px;color:#555;margin-bottom:5px">
        🔄 Line {done}/{total} &nbsp;·&nbsp; {segments} segment{'s' if segments != 1 else ''} processed
      </div>
      <div style="background:#e0e0e0;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:#4A90D9;height:100%;width:{pct}%;transition:width 0.3s ease;border-radius:6px"></div>
      </div>
    </div>
    """


def _cancelled_html(saved: int = 0) -> str:
    note = f" &nbsp;·&nbsp; {saved} file{'s' if saved != 1 else ''} saved" if saved else ""
    return f"""
    <div style="padding:6px 0">
      <div style="font-size:13px;color:#b26a00;margin-bottom:5px">
        ⏹ Cancelled{note}
      </div>
      <div style="background:#e0e0e0;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:#f0a500;height:100%;width:100%;border-radius:6px"></div>
      </div>
    </div>
    """


def _done_html(total: int) -> str:
    return f"""
    <div style="padding:6px 0">
      <div style="font-size:13px;color:#2e7d32;margin-bottom:5px">
        ✅ Done &nbsp;·&nbsp; {total} segment{'s' if total != 1 else ''}
      </div>
      <div style="background:#e0e0e0;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:#4CAF50;height:100%;width:100%;border-radius:6px"></div>
      </div>
    </div>
    """


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
