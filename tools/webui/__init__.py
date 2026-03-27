from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.inference import cancel_generation
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


# ── GPU monitor ────────────────────────────────────────────────────────────────
def _get_gpu_html() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
             "memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            raise RuntimeError()
        cards = []
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 8:
                continue
            idx, name, temp, gpu_pct, mem_used, mem_total, power, plimit = p[:8]
            gpu_f   = float(gpu_pct)  if gpu_pct  not in ("[N/A]", "N/A") else 0.0
            mem_u   = float(mem_used) if mem_used  not in ("[N/A]", "N/A") else 0.0
            mem_t   = float(mem_total)if mem_total not in ("[N/A]", "N/A") else 1.0
            temp_f  = float(temp)     if temp      not in ("[N/A]", "N/A") else 0.0
            pwr_f   = float(power)    if power     not in ("[N/A]", "N/A") else 0.0
            mem_pct = mem_u / mem_t * 100

            c_gpu  = "#ff4d6d" if gpu_f  > 90 else "#f59e0b" if gpu_f  > 70 else "#00e5a8"
            c_mem  = "#ff4d6d" if mem_pct > 90 else "#f59e0b" if mem_pct > 75 else "#00e5a8"
            c_temp = "#ff4d6d" if temp_f  > 85 else "#f59e0b" if temp_f  > 75 else "#00e5a8"

            cards.append(f"""
<div class="gpu-card">
  <div class="gpu-name">{name}</div>
  <div class="gpu-stats">
    <div class="gpu-stat">
      <span class="stat-label">GPU</span>
      <div class="stat-bar"><div class="stat-fill" style="width:{gpu_f:.0f}%;background:{c_gpu}"></div></div>
      <span class="stat-val" style="color:{c_gpu}">{gpu_f:.0f}%</span>
    </div>
    <div class="gpu-stat">
      <span class="stat-label">VRAM</span>
      <div class="stat-bar"><div class="stat-fill" style="width:{mem_pct:.0f}%;background:{c_mem}"></div></div>
      <span class="stat-val" style="color:{c_mem}">{mem_u/1024:.1f}/{mem_t/1024:.0f}G</span>
    </div>
    <div class="gpu-pill" style="color:{c_temp}">🌡 {temp_f:.0f}°C</div>
    <div class="gpu-pill">⚡ {pwr_f:.0f}W</div>
  </div>
</div>""")
        return "\n".join(cards) if cards else _gpu_na()
    except Exception:
        return _gpu_na()


def _gpu_na() -> str:
    return '<div class="gpu-card gpu-na">GPU monitor unavailable</div>'


# ── Dark theme ─────────────────────────────────────────────────────────────────
DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    # Body
    body_background_fill="#060810",
    body_text_color="#e2eaf6",
    body_text_color_subdued="#64748b",

    # Backgrounds
    background_fill_primary="#0d1321",
    background_fill_secondary="#111827",

    # Borders
    border_color_primary="rgba(148,163,184,0.09)",
    border_color_accent="#00e5a8",
    border_color_accent_subdued="rgba(0,229,168,0.3)",

    # Buttons — primary
    button_primary_background_fill="#00e5a8",
    button_primary_background_fill_hover="#00ffbe",
    button_primary_text_color="#060810",
    button_primary_border_color="#00e5a8",
    button_primary_border_color_hover="#00ffbe",

    # Buttons — secondary
    button_secondary_background_fill="#1a2233",
    button_secondary_background_fill_hover="#1e2d3d",
    button_secondary_text_color="#94a3b8",
    button_secondary_border_color="rgba(148,163,184,0.12)",

    # Buttons — stop
    button_cancel_background_fill="rgba(255,77,109,0.12)",
    button_cancel_background_fill_hover="#ff4d6d",
    button_cancel_text_color="#ff4d6d",
    button_cancel_border_color="rgba(255,77,109,0.25)",

    # Inputs
    input_background_fill="#080c16",
    input_background_fill_focus="#0a0e18",
    input_border_color="rgba(148,163,184,0.12)",
    input_border_color_focus="#00e5a8",
    input_placeholder_color="#334155",
    input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(0,229,168,0.08)",

    # Slider
    slider_color="#00e5a8",
    slider_color_dark="#00e5a8",

    # Blocks
    block_background_fill="#0d1321",
    block_border_color="rgba(148,163,184,0.09)",
    block_border_width="1px",
    block_label_background_fill="#111827",
    block_label_text_color="#64748b",
    block_title_text_color="#e2eaf6",
    block_shadow="none",

    # Panels
    panel_background_fill="#080c16",
    panel_border_color="rgba(148,163,184,0.09)",

    # Checkbox / radio
    checkbox_background_color="#1a2233",
    checkbox_background_color_selected="#00e5a8",
    checkbox_border_color="rgba(148,163,184,0.2)",
    checkbox_border_color_selected="#00e5a8",
    checkbox_label_background_fill="#111827",
    checkbox_label_background_fill_hover="#1a2233",
    checkbox_label_background_fill_selected="rgba(0,229,168,0.1)",
    checkbox_label_border_color="rgba(148,163,184,0.09)",
    checkbox_label_border_color_selected="#00e5a8",
    checkbox_label_text_color="#94a3b8",
    checkbox_label_text_color_selected="#00e5a8",

    # Errors
    error_background_fill="rgba(255,77,109,0.08)",
    error_border_color="rgba(255,77,109,0.2)",
    error_text_color="#ff4d6d",
    error_icon_color="#ff4d6d",

    # Shadows
    shadow_drop="none",
    shadow_drop_lg="none",
    shadow_spread="none",
)


# ── CSS — only for things the theme can't reach ────────────────────────────────
CUSTOM_CSS = """
/* GPU monitor card */
.gpu-card {
  background: #0d1321;
  border: 1px solid rgba(148,163,184,0.09);
  border-radius: 10px;
  padding: 12px 14px;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 8px;
}
.gpu-card:last-child { margin-bottom: 0; }
.gpu-na { color: #334155; font-size: 11px; }
.gpu-name {
  font-size: 12px; font-weight: 600;
  color: #e2eaf6; margin-bottom: 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.gpu-stats { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.gpu-stat  { display: flex; align-items: center; gap: 6px; flex: 1 1 120px; }
.stat-label { font-size: 10px; color: #475569; letter-spacing: 0.06em; width: 32px; flex-shrink: 0; }
.stat-bar   { flex: 1; height: 4px; background: #1a2233; border-radius: 2px; overflow: hidden; }
.stat-fill  { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.stat-val   { font-size: 11px; font-weight: 600; width: 48px; text-align: right; flex-shrink: 0; }
.gpu-pill   { font-size: 11px; color: #94a3b8; background: #1a2233;
              padding: 2px 8px; border-radius: 99px; white-space: nowrap; flex-shrink: 0; }

/* App header */
.fish-header {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 20px;
  background: #0d1321;
  border-bottom: 1px solid rgba(148,163,184,0.09);
  margin-bottom: 16px;
}
.fish-header-icon {
  width: 34px; height: 34px; border-radius: 9px;
  background: #00e5a8;
  display: flex; align-items: center; justify-content: center;
  font-size: 17px; flex-shrink: 0;
  box-shadow: 0 0 14px rgba(0,229,168,0.3);
}
.fish-header-text { display: flex; flex-direction: column; }
.fish-title {
  font-size: 16px; font-weight: 700; color: #e2eaf6;
  font-family: 'DM Sans', sans-serif; line-height: 1.2;
  letter-spacing: -0.02em;
}
.fish-sub { font-size: 11px; color: #475569; margin-top: 1px; }
.fish-badge {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
  color: #00e5a8; background: rgba(0,229,168,0.08);
  border: 1px solid rgba(0,229,168,0.25);
  padding: 3px 9px; border-radius: 99px;
}

/* Progress HTML blocks — remove box when empty */
.status-area { min-height: 0 !important; }
.status-area > div:empty { display: none; }
.status-area .html-container:empty { display: none; }

/* Sliders — cleaner track */
input[type=range]::-webkit-slider-thumb {
  box-shadow: 0 0 6px rgba(0,229,168,0.5) !important;
}

/* Generate / Stop button row */
.action-row { display: flex !important; gap: 10px !important; }
.action-row > button:first-child {
  font-size: 15px !important; font-weight: 700 !important; letter-spacing: 0.01em !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c16; }
::-webkit-scrollbar-thumb { background: #1a2233; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
"""


# ── App builder ────────────────────────────────────────────────────────────────
def build_app(inference_fct: Callable, engine=None, theme: str = "dark") -> gr.Blocks:
    with gr.Blocks(title="Fish Speech") as app:

        # Header
        gr.HTML("""
        <div class="fish-header">
          <div class="fish-header-icon">🐟</div>
          <div class="fish-header-text">
            <div class="fish-title">Fish Speech</div>
            <div class="fish-sub">Neural Text-to-Speech</div>
          </div>
          <div class="fish-badge">S2-PRO</div>
        </div>
        """)

        # Force dark theme URL param
        app.load(None, None, js=(
            "() => { const p = new URLSearchParams(window.location.search);"
            " if (!p.has('__theme')) { p.set('__theme','dark'); window.location.search = p.toString(); } }"
        ))

        # ── Two-column layout ────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left: input + settings ───────────────────────────────────────
            with gr.Column(scale=11):

                text = gr.Textbox(
                    label=i18n("Input Text"),
                    placeholder=TEXTBOX_PLACEHOLDER,
                    lines=11,
                )

                with gr.Tabs():

                    # ── Generation settings ──────────────────────────────────
                    with gr.Tab("⚙  Generation"):
                        with gr.Row():
                            chunk_length = gr.Slider(
                                label=i18n("Chunk Length"),
                                info="Tokens per iterative chunk · 0 = off",
                                minimum=100, maximum=800, value=600, step=8,
                            )
                            max_new_tokens = gr.Slider(
                                label=i18n("Max Tokens / Batch"),
                                info="0 = no limit",
                                minimum=0, maximum=2048, value=0, step=8,
                            )
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                info="Higher = more expressive",
                                minimum=0.1, maximum=1.0, value=0.75, step=0.01,
                            )
                            top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.85, step=0.01,
                            )
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label=i18n("Repetition Penalty"),
                                minimum=1.0, maximum=1.5, value=1.05, step=0.01,
                            )
                            seed = gr.Number(
                                label="Seed",
                                info="0 = random",
                                value=0,
                            )

                    # ── Reference voice ──────────────────────────────────────
                    with gr.Tab("🎙  Reference Voice"):
                        gr.Markdown(
                            "Upload **5–10 s** of clean speech to clone the voice, "
                            "or pick a saved voice from the dropdown."
                        )
                        with gr.Row():
                            use_memory_cache = gr.Radio(
                                label=i18n("Memory Cache"),
                                choices=["on", "off"], value="on",
                            )
                        with gr.Row():
                            reference_id = gr.Dropdown(
                                label=i18n("Saved Voice"),
                                choices=[""] + (engine.list_reference_ids() if engine else []),
                                value="",
                                info="Leave empty to use uploaded audio below",
                                allow_custom_value=True,
                            )
                            refresh_btn = gr.Button(
                                "↻", scale=0, min_width=44, variant="secondary",
                            )
                        reference_audio = gr.Audio(
                            label=i18n("Reference Audio"), type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label=i18n("Reference Transcript"), lines=2,
                            placeholder="Type the exact words spoken in the reference clip…",
                        )
                        gr.Markdown("---\n**Save as named voice**")
                        with gr.Row():
                            save_voice_name = gr.Textbox(
                                label=i18n("Voice Name"),
                                placeholder="e.g. narrator", scale=3,
                            )
                            save_voice_btn = gr.Button(
                                i18n("Save Voice"), scale=1, variant="secondary",
                            )
                        save_voice_status = gr.HTML()

            # ── Right: output + controls ─────────────────────────────────────
            with gr.Column(scale=9):

                # GPU Monitor — updates every 2 s
                gpu_html = gr.HTML(value=_get_gpu_html, every=2)

                # Status area
                with gr.Group(elem_classes=["status-area"]):
                    error    = gr.HTML(visible=True)
                    progress = gr.HTML(value="", visible=True)

                # Audio output
                audio = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy", interactive=False,
                )

                # ZIP download
                zip_file = gr.File(
                    label=i18n("Download ZIP  (Line-by-line)"),
                    interactive=False,
                )

                gr.HTML("<div style='height:2px'></div>")

                # Mode
                mode = gr.Radio(
                    label=i18n("Generation Mode"),
                    choices=["Normal", "Line-by-line"],
                    value="Normal",
                    info="Line-by-line: each line → separate WAV, packed into ZIP",
                )

                gr.HTML("<div style='height:4px'></div>")

                # Action buttons
                with gr.Row(elem_classes=["action-row"]):
                    generate = gr.Button(
                        "▶  " + i18n("Generate"),
                        variant="primary", scale=3,
                    )
                    stop_btn = gr.Button(
                        "⏹  " + i18n("Stop"),
                        variant="stop", scale=1,
                    )

        # ── Event wiring ─────────────────────────────────────────────────────
        generate_event = generate.click(
            inference_fct,
            inputs=[
                text, mode,
                reference_id, reference_audio, reference_text,
                max_new_tokens, chunk_length,
                top_p, repetition_penalty, temperature, seed,
                use_memory_cache,
            ],
            outputs=[audio, error, progress, zip_file],
            concurrency_limit=1,
            api_name="generate",
        )

        stop_btn.click(fn=cancel_generation, cancels=[generate_event])

        def refresh_voices():
            ids = engine.list_reference_ids() if engine else []
            return gr.Dropdown(choices=[""] + ids)
        refresh_btn.click(refresh_voices, outputs=reference_id)

        def save_voice(audio_path, ref_text, voice_name):
            if not voice_name or not voice_name.strip():
                return "<div style='color:#ff4d6d'>Please enter a voice name.</div>"
            if not audio_path:
                return "<div style='color:#ff4d6d'>Please upload a reference audio file first.</div>"
            if not ref_text or not ref_text.strip():
                return "<div style='color:#ff4d6d'>Please enter the reference text first.</div>"
            if engine is None:
                return "<div style='color:#ff4d6d'>Engine not available.</div>"
            try:
                engine.add_reference(voice_name.strip(), audio_path, ref_text.strip())
                return f"<div style='color:#00e5a8'>✓ Voice <b>{voice_name.strip()}</b> saved.</div>"
            except FileExistsError:
                return f"<div style='color:#ff4d6d'>Name <b>{voice_name.strip()}</b> already exists.</div>"
            except Exception as e:
                return f"<div style='color:#ff4d6d'>Error: {e}</div>"

        save_voice_btn.click(
            save_voice,
            inputs=[reference_audio, reference_text, save_voice_name],
            outputs=[save_voice_status],
        )

    return app
