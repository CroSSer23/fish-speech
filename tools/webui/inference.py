import html
import io
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


def _crossfade_concat(segments: list, sample_rate: int, overlap_ms: int) -> np.ndarray:
    """Concatenate audio segments with crossfade overlap between consecutive lines."""
    if not segments:
        return np.array([], dtype=np.float32)

    n_cf = int(sample_rate * overlap_ms / 1000)
    result = segments[0].astype(np.float32)

    for seg in segments[1:]:
        seg = seg.astype(np.float32)

        if n_cf <= 0 or len(result) < n_cf or len(seg) < n_cf:
            result = np.concatenate([result, seg])
            continue

        fade_out = np.linspace(1.0, 0.0, n_cf, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, n_cf, dtype=np.float32)
        overlap  = result[-n_cf:] * fade_out + seg[:n_cf] * fade_in
        result   = np.concatenate([result[:-n_cf], overlap, seg[n_cf:]])

    return result


def _audio_to_bytes(data: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 numpy audio array to WAV bytes (for ServeReferenceAudio)."""
    buf = io.BytesIO()
    sf.write(buf, data.astype(np.float32), sample_rate, format="WAV")
    return buf.getvalue()


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
    extra_references=None,
) -> ServeTTSRequest:
    references = list(extra_references) if extra_references else []
    if reference_audio:
        references += get_reference_audio(reference_audio, reference_text)

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
    overlap_ms,
    context_chars,
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
        yield from _inference_linewise(text, engine, overlap_ms=int(overlap_ms), context_chars=int(context_chars), **common)
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

def _inference_linewise(text, engine, overlap_ms=80, context_chars=30, **common):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        yield None, build_html_error_message(Exception("No lines to process")), "", None
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    raw_segments = []   # float32 arrays — for crossfade concat
    errors = []
    total_segments = 0
    sample_rate_out = None
    prev_audio_bytes: bytes | None = None   # previous segment encoded as WAV bytes
    prev_line_text: str = ""                # text of previous segment (for reference transcript)

    for idx, line in enumerate(lines, start=1):
        if _cancel_event.is_set():
            break
        print(f"[line-by-line] {idx}/{len(lines)}: {line[:60]}...")
        yield gr.update(), gr.update(), _line_progress_html(idx - 1, len(lines), total_segments), gr.update()

        # ── Prosody context via audio reference ────────────────────────────────
        # Pass the previous segment's audio as ServeReferenceAudio so the model
        # hears how the previous line sounded and continues with matching prosody.
        # Unlike text-prefix + trimming, this never repeats words in the output.
        extra_refs = []
        if context_chars > 0 and prev_audio_bytes is not None:
            ref_text = prev_line_text[-context_chars:].strip()
            extra_refs = [ServeReferenceAudio(audio=prev_audio_bytes, text=ref_text)]

        line_segments = 0
        try:
            req = _make_request(text=line, extra_references=extra_refs, **common)
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
        sample_rate_out = sample_rate
        data_f32 = data.astype(np.float32)

        # Encode for next iteration's reference (before fades, to keep it clean)
        prev_audio_bytes = _audio_to_bytes(data_f32, sample_rate)
        prev_line_text = line

        data_faded = _apply_fades(data_f32, sample_rate)

        saved = _auto_save((sample_rate, data_faded), batch_dir / f"{idx}.wav")
        if saved:
            print(f"[auto-save] Saved: {saved}")
            saved_paths.append(saved)

        raw_segments.append(data_f32)

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

    # Build combined audio with crossfade overlap between lines
    combined_audio = None
    if raw_segments and sample_rate_out is not None:
        combined = _crossfade_concat(raw_segments, sample_rate_out, overlap_ms)
        combined_i16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
        combined_audio = (sample_rate_out, combined_i16)
        combined_path = batch_dir.parent / f"{timestamp}_combined.wav"
        _auto_save((sample_rate_out, combined), combined_path)
        print(f"[line-by-line] Combined audio saved: {combined_path}")

    error_html = build_html_error_message(Exception("; ".join(errors))) if errors else None
    status_html = _cancelled_html(len(saved_paths)) if cancelled else _done_html(total_segments)
    yield combined_audio, error_html, status_html, str(zip_path)


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
