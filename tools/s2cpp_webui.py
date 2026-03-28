"""
Standalone Gradio web UI for Fish Speech using s2.cpp GGUF backend.
Designed for machines with limited VRAM (< 20 GB).
No PyTorch dependency — uses s2.cpp binary for inference.
"""
import argparse
import subprocess
import tempfile
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

OUTPUT_DIR = Path("outputs")
FADE_MS = 80

_cancel_event = threading.Event()


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _apply_fades(data: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fade = min(int(sample_rate * FADE_MS / 1000), len(data) // 4)
    if n_fade <= 0:
        return data
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    out = data.copy().astype(np.float32)
    out[:n_fade] *= ramp
    out[-n_fade:] *= ramp[::-1]
    return out


def _find_s2_binary(s2cpp_dir: Path) -> Path:
    """Find the compiled s2 binary across possible CMake output locations."""
    candidates = [
        s2cpp_dir / "build" / "Release" / "s2.exe",  # MSVC
        s2cpp_dir / "build" / "s2.exe",               # MinGW Windows
        s2cpp_dir / "build" / "Release" / "s2",       # MSVC Linux-style
        s2cpp_dir / "build" / "s2",                    # Unix/MinGW
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"s2 binary not found in {s2cpp_dir}/build. "
        "Build s2.cpp first with cmake."
    )


# ── s2.cpp inference ──────────────────────────────────────────────────────────

def _run_s2cpp(
    text: str,
    s2_bin: Path,
    model_path: Path,
    tokenizer_path: Path,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    temperature: float = 0.75,
    top_p: float = 0.85,
    max_tokens: int = 0,
) -> tuple[int, np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        str(s2_bin),
        "-m", str(model_path),
        "-t", str(tokenizer_path),
        "-c", "0",
        "-text", text,
        "-o", str(out_path),
        "-temp", str(temperature),
        "-top-p", str(top_p),
        "--normalize",
        "--trim-silence",
    ]
    if max_tokens > 0:
        cmd += ["-max-tokens", str(max_tokens)]
    if reference_audio:
        cmd += ["-pa", str(reference_audio)]
    if reference_text:
        cmd += ["-pt", reference_text]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "s2.cpp returned non-zero exit code")

    data, sr = sf.read(str(out_path))
    out_path.unlink(missing_ok=True)
    return sr, data.astype(np.float32)


# ── Inference generators ───────────────────────────────────────────────────────

def cancel_generation():
    _cancel_event.set()


def _progress_html(msg: str, pct: int, color: str = "#00e5a8") -> str:
    return f"""
    <div style="padding:6px 0">
      <div style="font-size:13px;color:#555;margin-bottom:5px">{msg}</div>
      <div style="background:#e0e0e0;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:{color};height:100%;width:{pct}%;transition:width 0.3s ease;border-radius:6px"></div>
      </div>
    </div>"""


def _inference_normal(text, s2_bin, model_path, tokenizer_path,
                      ref_audio, ref_text, temperature, top_p, max_tokens):
    try:
        yield gr.update(), gr.update(), _progress_html("⏳ Generating…", 50), gr.update()

        if _cancel_event.is_set():
            yield None, None, _progress_html("⏹ Cancelled", 100, "#f0a500"), None
            return

        sr, data = _run_s2cpp(text, s2_bin, model_path, tokenizer_path,
                              ref_audio, ref_text, temperature, top_p, max_tokens)

        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        sf.write(str(OUTPUT_DIR / f"{ts}.wav"), data, sr)

        data_i16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        yield (sr, data_i16), None, _progress_html("✅ Done", 100, "#4CAF50"), None

    except Exception as e:
        yield None, f'<div style="color:#ff4d6d;font-weight:bold">{e}</div>', "", None


def _inference_linewise(text, s2_bin, model_path, tokenizer_path,
                        ref_audio, ref_text, temperature, top_p, max_tokens):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        yield None, '<div style="color:#ff4d6d">No lines to process</div>', "", None
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / ts
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved_paths, errors = [], []

    for idx, line in enumerate(lines, 1):
        if _cancel_event.is_set():
            break
        pct = int((idx - 1) / len(lines) * 100)
        yield gr.update(), gr.update(), _progress_html(
            f"⏳ Line {idx - 1}/{len(lines)}", pct), gr.update()

        try:
            sr, data = _run_s2cpp(line, s2_bin, model_path, tokenizer_path,
                                  ref_audio, ref_text, temperature, top_p, max_tokens)
            faded = _apply_fades(data, sr)
            out_path = batch_dir / f"{idx}.wav"
            sf.write(str(out_path), faded, sr)
            saved_paths.append(out_path)
        except Exception as e:
            errors.append(f"Line {idx}: {e}")

    cancelled = _cancel_event.is_set()

    if not saved_paths:
        msg = "Cancelled — no files saved." if cancelled else "; ".join(errors) or "No audio generated."
        yield None, f'<div style="color:#ff4d6d">{msg}</div>', "", None
        return

    zip_path = OUTPUT_DIR / f"{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in saved_paths:
            zf.write(p, p.name)

    status_color = "#f0a500" if cancelled else "#4CAF50"
    status_icon = "⏹ Cancelled" if cancelled else "✅ Done"
    status = _progress_html(f"{status_icon} · {len(saved_paths)} files", 100, status_color)
    err_html = f'<div style="color:#ff4d6d">{"; ".join(errors)}</div>' if errors else None
    yield None, err_html, status, str(zip_path)


def inference_wrapper(text, mode, ref_audio, ref_text,
                      max_tokens, temperature, top_p, seed,
                      s2_bin, model_path, tokenizer_path):
    _cancel_event.clear()
    kwargs = dict(
        s2_bin=s2_bin, model_path=model_path, tokenizer_path=tokenizer_path,
        ref_audio=ref_audio or None,
        ref_text=ref_text or None,
        temperature=temperature, top_p=top_p, max_tokens=int(max_tokens),
    )
    if mode == "Line-by-line":
        yield from _inference_linewise(text, **kwargs)
    else:
        yield from _inference_normal(text, **kwargs)


# ── Dark theme (self-contained, no fish_speech import) ────────────────────────

DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#060810",
    body_text_color="#e2eaf6",
    background_fill_primary="#0d1321",
    background_fill_secondary="#111827",
    border_color_primary="rgba(148,163,184,0.09)",
    button_primary_background_fill="#00e5a8",
    button_primary_background_fill_hover="#00ffbe",
    button_primary_text_color="#060810",
    button_secondary_background_fill="#1a2233",
    button_secondary_text_color="#94a3b8",
    button_cancel_background_fill="rgba(255,77,109,0.12)",
    button_cancel_background_fill_hover="#ff4d6d",
    button_cancel_text_color="#ff4d6d",
    button_cancel_border_color="rgba(255,77,109,0.25)",
    input_background_fill="#080c16",
    input_border_color_focus="#00e5a8",
    slider_color="#00e5a8",
    block_background_fill="#0d1321",
    block_border_color="rgba(148,163,184,0.09)",
)

CUSTOM_CSS = """
.fish-header {
  display:flex;align-items:center;gap:12px;
  padding:14px 20px;background:#0d1321;
  border-bottom:1px solid rgba(148,163,184,0.09);margin-bottom:16px;
}
.fish-header-icon {
  width:34px;height:34px;border-radius:9px;background:#00e5a8;
  display:flex;align-items:center;justify-content:center;
  font-size:17px;box-shadow:0 0 14px rgba(0,229,168,0.3);
}
.fish-title{font-size:16px;font-weight:700;color:#e2eaf6;letter-spacing:-0.02em;}
.fish-sub{font-size:11px;color:#475569;}
.fish-badge{
  margin-left:auto;font-family:'JetBrains Mono',monospace;
  font-size:10px;font-weight:600;color:#00e5a8;
  background:rgba(0,229,168,0.08);border:1px solid rgba(0,229,168,0.25);
  padding:3px 9px;border-radius:99px;
}
footer{display:none!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#080c16;}
::-webkit-scrollbar-thumb{background:#1a2233;border-radius:99px;}
"""


# ── App builder ────────────────────────────────────────────────────────────────

def build_app(s2_bin: Path, model_path: Path, tokenizer_path: Path) -> gr.Blocks:
    from functools import partial
    infer = partial(inference_wrapper,
                    s2_bin=s2_bin, model_path=model_path, tokenizer_path=tokenizer_path)

    with gr.Blocks(title="Fish Speech · GGUF") as app:
        gr.HTML("""
        <div class="fish-header">
          <div class="fish-header-icon">🐟</div>
          <div class="fish-header-text">
            <div class="fish-title">Fish Speech</div>
            <div class="fish-sub">s2.cpp · GGUF backend</div>
          </div>
          <div class="fish-badge">GGUF</div>
        </div>""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=11):
                text = gr.Textbox(label="Input Text", lines=11,
                                  placeholder="Enter text to synthesize…")
                with gr.Tabs():
                    with gr.Tab("⚙  Generation"):
                        with gr.Row():
                            temperature = gr.Slider(label="Temperature", minimum=0.1,
                                                    maximum=1.0, value=0.75, step=0.01)
                            top_p = gr.Slider(label="Top-P", minimum=0.1,
                                              maximum=1.0, value=0.85, step=0.01)
                        with gr.Row():
                            max_tokens = gr.Number(label="Max Tokens (0 = auto)", value=0)
                            seed = gr.Number(label="Seed (0 = random)", value=0)
                    with gr.Tab("🎙  Reference Voice"):
                        gr.Markdown("Upload **5–10 s** of clean speech to clone the voice.")
                        ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                        ref_text = gr.Textbox(label="Reference Transcript", lines=2,
                                              placeholder="Exact words spoken in the clip…")

            with gr.Column(scale=9):
                error = gr.HTML()
                progress = gr.HTML(value="")
                audio = gr.Audio(label="Generated Audio", type="numpy", interactive=False)
                zip_file = gr.File(label="Download ZIP  (Line-by-line)", interactive=False)
                mode = gr.Radio(label="Generation Mode",
                                choices=["Normal", "Line-by-line"], value="Normal",
                                info="Line-by-line: each line → separate WAV → ZIP")
                with gr.Row():
                    generate = gr.Button("▶  Generate", variant="primary", scale=3)
                    stop_btn = gr.Button("⏹  Stop", variant="stop", scale=1)

        generate_event = generate.click(
            infer,
            inputs=[text, mode, ref_audio, ref_text, max_tokens, temperature, top_p, seed],
            outputs=[audio, error, progress, zip_file],
            api_name="generate",
            concurrency_limit=1,
        )
        stop_btn.click(fn=cancel_generation, cancels=[generate_event])

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Speech GGUF Web UI via s2.cpp")
    parser.add_argument("--s2cpp-dir", type=Path, default=Path("../s2cpp"),
                        help="Root directory of the compiled s2.cpp repo")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to .gguf model file (auto-detected if omitted)")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to tokenizer.json (auto-detected if omitted)")
    args = parser.parse_args()

    s2_bin = _find_s2_binary(args.s2cpp_dir)
    print(f"s2 binary: {s2_bin}")

    model_path = args.model or next(args.s2cpp_dir.glob("models/*.gguf"), None)
    if not model_path or not model_path.exists():
        raise FileNotFoundError("No .gguf model found. Run install first.")

    tokenizer_path = args.tokenizer or (args.s2cpp_dir / "models" / "tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")

    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")

    app = build_app(s2_bin, model_path, tokenizer_path)
    app.launch(share=True, theme=DARK_THEME, css=CUSTOM_CSS, server_name="0.0.0.0")
