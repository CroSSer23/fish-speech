from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.inference import cancel_generation
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


# ── Dark Studio CSS ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ────────────────────────────────────────────── */
:root {
  --bg-void:    #060810;
  --bg-base:    #0a0e18;
  --bg-card:    #0f1520;
  --bg-panel:   #141b28;
  --bg-raised:  #1a2233;
  --bg-hover:   #1e293b;
  --border:     rgba(148,163,184,0.07);
  --border-md:  rgba(148,163,184,0.12);
  --border-hi:  rgba(0,229,168,0.35);
  --accent:     #00e5a8;
  --accent-dim: #00a876;
  --accent-glow:rgba(0,229,168,0.18);
  --accent-xglow:rgba(0,229,168,0.08);
  --red:        #ff4d6d;
  --red-soft:   rgba(255,77,109,0.15);
  --orange:     #f59e0b;
  --text-1:     #e8eef8;
  --text-2:     #7d8fa8;
  --text-3:     #3d4f65;
  --sans:       'DM Sans', -apple-system, sans-serif;
  --display:    'Syne', sans-serif;
  --mono:       'JetBrains Mono', 'Fira Code', monospace;
  --r:          10px;
  --r-sm:       7px;
  --t:          0.16s cubic-bezier(0.4,0,0.2,1);
  --shadow:     0 4px 24px rgba(0,0,0,0.5);
}

/* ── Base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
gradio-app,
#root,
.app {
  background: var(--bg-void) !important;
  font-family: var(--sans) !important;
  color: var(--text-1) !important;
  min-height: 100vh;
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}

footer, .footer { display: none !important; }

/* ── Scrollbars ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-raised); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ── App header ────────────────────────────────────────────────── */
.app-header {
  background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-base) 100%);
  border-bottom: 1px solid var(--border-md);
  padding: 18px 28px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.app-header-icon {
  width: 36px; height: 36px;
  background: var(--accent);
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  box-shadow: 0 0 16px var(--accent-glow);
  flex-shrink: 0;
}
.app-header-title {
  font-family: var(--display);
  font-size: 20px; font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.app-header-sub {
  font-size: 12px; color: var(--text-2);
  font-weight: 400; margin-top: 2px;
}
.app-header-badge {
  margin-left: auto;
  background: var(--accent-xglow);
  border: 1px solid var(--border-hi);
  color: var(--accent);
  font-size: 11px; font-weight: 600;
  font-family: var(--mono);
  padding: 3px 10px;
  border-radius: 99px;
  letter-spacing: 0.04em;
}

/* ── Section labels ────────────────────────────────────────────── */
.section-label {
  font-family: var(--mono);
  font-size: 10px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-3);
  padding: 14px 18px 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0;
}

/* ── Panels / Blocks ───────────────────────────────────────────── */
.block, .panel, .form, .box,
.wrap.svelte-1ipelgc, .wrap.svelte-z7cif2 {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  box-shadow: none !important;
}

/* ── Row / Column gaps ─────────────────────────────────────────── */
.gap, .gap-2, .flex {
  gap: 10px !important;
}
.contain { padding: 16px !important; }
.padded { padding: 14px 16px !important; }

/* ── Prose / Markdown ──────────────────────────────────────────── */
.prose, .prose *, .markdown-body, .markdown-body * {
  color: var(--text-1) !important;
  font-family: var(--sans) !important;
}
.prose h1, .markdown h1 {
  font-family: var(--display) !important;
  font-size: 22px !important; font-weight: 700 !important;
  color: var(--text-1) !important;
  letter-spacing: -0.03em !important;
}
.prose h2, .prose h3 {
  font-family: var(--display) !important;
  color: var(--text-1) !important;
}
.prose p { color: var(--text-2) !important; font-size: 13px !important; }
.prose a { color: var(--accent) !important; }
.prose hr { border-color: var(--border-md) !important; margin: 12px 0 !important; }
.prose strong, b { color: var(--text-1) !important; }

/* ── Component labels ──────────────────────────────────────────── */
label > span,
.label-wrap > span,
span.svelte-1gfkn6j,
.block > label > span {
  color: var(--text-2) !important;
  font-size: 11px !important; font-weight: 500 !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
  font-family: var(--mono) !important;
  margin-bottom: 6px !important;
  display: block !important;
}

/* ── Textareas & Text inputs ───────────────────────────────────── */
textarea,
input[type="text"],
input[type="number"],
input[type="email"],
input[type="search"] {
  background: var(--bg-base) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-1) !important;
  font-family: var(--sans) !important;
  font-size: 14px !important; line-height: 1.6 !important;
  padding: 10px 13px !important;
  transition: border-color var(--t), box-shadow var(--t) !important;
  caret-color: var(--accent) !important;
  resize: vertical !important;
}
textarea:focus, input:focus {
  border-color: var(--border-hi) !important;
  box-shadow: 0 0 0 3px var(--accent-xglow) !important;
  outline: none !important;
  background: var(--bg-card) !important;
}
textarea::placeholder, input::placeholder {
  color: var(--text-3) !important;
  font-style: italic !important;
}
textarea:hover:not(:focus), input:hover:not(:focus) {
  border-color: var(--border-md) !important;
}

/* ── Number input ──────────────────────────────────────────────── */
.number input { text-align: center !important; }

/* ── Sliders ───────────────────────────────────────────────────── */
.wrap input[type="range"],
input[type="range"] {
  -webkit-appearance: none !important; appearance: none !important;
  height: 3px !important; border-radius: 99px !important;
  background: var(--bg-raised) !important;
  border: none !important; outline: none !important;
  width: 100% !important; cursor: pointer !important;
}
input[type="range"]::-webkit-slider-track {
  height: 3px !important; border-radius: 99px !important;
  background: var(--bg-raised) !important;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none !important; appearance: none !important;
  width: 15px !important; height: 15px !important;
  border-radius: 50% !important;
  background: var(--accent) !important;
  box-shadow: 0 0 8px var(--accent-glow) !important;
  cursor: pointer !important;
  transition: transform var(--t), box-shadow var(--t) !important;
  margin-top: -6px !important;
}
input[type="range"]:hover::-webkit-slider-thumb {
  transform: scale(1.25) !important;
  box-shadow: 0 0 14px var(--accent-glow) !important;
}
input[type="range"]::-moz-range-thumb {
  width: 15px !important; height: 15px !important;
  border-radius: 50% !important; border: none !important;
  background: var(--accent) !important;
  box-shadow: 0 0 8px var(--accent-glow) !important;
}
/* slider value label */
.wrap .output-class, .value-text, .max-w-full.svelte-1r56dez {
  color: var(--accent) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important; font-weight: 500 !important;
}

/* ── Buttons ───────────────────────────────────────────────────── */
button {
  font-family: var(--sans) !important;
  font-weight: 600 !important; letter-spacing: 0.01em !important;
  border-radius: var(--r-sm) !important;
  transition: all var(--t) !important;
  cursor: pointer !important;
  font-size: 14px !important;
}

/* Primary / Generate */
button.primary,
button[data-testid*="generate"],
.primary > button {
  background: var(--accent) !important;
  border: none !important;
  color: #060810 !important;
  padding: 11px 22px !important;
  font-size: 14px !important; font-weight: 700 !important;
  box-shadow: 0 0 0 0 var(--accent-glow) !important;
  letter-spacing: 0.02em !important;
}
button.primary:hover {
  background: #00ffc0 !important;
  box-shadow: 0 4px 20px var(--accent-glow), 0 0 30px var(--accent-xglow) !important;
  transform: translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }
button.primary:disabled {
  background: var(--bg-raised) !important;
  color: var(--text-3) !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
  transform: none !important;
}

/* Stop */
button.stop {
  background: var(--red-soft) !important;
  border: 1px solid rgba(255,77,109,0.25) !important;
  color: var(--red) !important;
  padding: 11px 18px !important;
}
button.stop:hover {
  background: var(--red) !important;
  border-color: var(--red) !important;
  color: white !important;
  box-shadow: 0 4px 16px rgba(255,77,109,0.4) !important;
  transform: translateY(-1px) !important;
}

/* Secondary */
button.secondary {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-md) !important;
  color: var(--text-2) !important;
  padding: 8px 14px !important;
}
button.secondary:hover {
  border-color: var(--border-hi) !important;
  color: var(--accent) !important;
  background: var(--bg-hover) !important;
}

/* Icon / small buttons */
button.sm { padding: 6px 12px !important; font-size: 12px !important; }
button[aria-label="↻"], .refresh-btn {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border-md) !important;
  color: var(--text-2) !important;
  min-width: 40px !important; padding: 8px !important;
}
button[aria-label="↻"]:hover { color: var(--accent) !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
.tabs { background: transparent !important; }
.tab-nav, div[role="tablist"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: 4px !important; gap: 2px !important;
  margin-bottom: 12px !important;
}
.tab-nav button, div[role="tablist"] button {
  background: transparent !important;
  border: none !important;
  color: var(--text-2) !important;
  padding: 7px 16px !important;
  border-radius: 6px !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  transition: all var(--t) !important;
}
.tab-nav button.selected, div[role="tablist"] button[aria-selected="true"] {
  background: var(--accent-glow) !important;
  color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--border-hi) !important;
}
.tab-nav button:hover:not(.selected) {
  background: var(--bg-hover) !important;
  color: var(--text-1) !important;
}

/* ── Radio groups ──────────────────────────────────────────────── */
.wrap.svelte-1p9xokt { background: transparent !important; border: none !important; }
fieldset { border: none !important; padding: 0 !important; margin: 0 !important; }
.wrap label, .radio-group label {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 6px !important;
  padding: 7px 14px !important;
  cursor: pointer !important;
  transition: all var(--t) !important;
  color: var(--text-2) !important;
  font-size: 13px !important; font-weight: 500 !important;
}
.wrap label:has(input:checked), .radio-group label:has(input:checked) {
  background: var(--accent-glow) !important;
  border-color: var(--border-hi) !important;
  color: var(--accent) !important;
}
.wrap label:hover:not(:has(input:checked)) {
  border-color: var(--border-md) !important;
  background: var(--bg-hover) !important;
  color: var(--text-1) !important;
}
input[type="radio"] { accent-color: var(--accent) !important; }

/* ── Dropdown / Select ─────────────────────────────────────────── */
.wrap-inner, .wrap.svelte-1ipelgc {
  background: var(--bg-base) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: var(--r-sm) !important;
}
select, .dropdown, .select-wrap {
  background: var(--bg-base) !important;
  border: 1px solid var(--border-md) !important;
  color: var(--text-1) !important;
  border-radius: var(--r-sm) !important;
}
.token.svelte-dpvpv {
  background: var(--accent-xglow) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 4px !important;
}
.item.svelte-dpvpv { color: var(--text-1) !important; }
.item.svelte-dpvpv:hover { background: var(--bg-hover) !important; }
ul.options.svelte-dpvpv {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: var(--r-sm) !important;
}

/* ── Audio player ──────────────────────────────────────────────── */
.waveform-container, audio-player, .audio-container, .audio {
  background: var(--bg-base) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}
audio {
  border-radius: var(--r-sm) !important;
  background: var(--bg-base) !important;
  filter: invert(0) !important;
}
.waveform {
  background: var(--bg-base) !important;
}
/* WaveSurfer waveform bars - make teal */
wave canvas { filter: hue-rotate(145deg) saturate(1.5) !important; }

/* ── File upload ───────────────────────────────────────────────── */
.file-preview, .file-preview-holder,
.upload-container, .upload-btn-container {
  background: var(--bg-base) !important;
  border: 1px dashed var(--border-md) !important;
  border-radius: var(--r) !important;
  color: var(--text-2) !important;
}
.file-preview:hover {
  border-color: var(--border-hi) !important;
  background: var(--bg-card) !important;
}
.upload-btn { color: var(--accent) !important; }

/* ── HTML output (progress bars, errors) ───────────────────────── */
.html-container { background: transparent !important; }
.html-container > div {
  font-family: var(--sans) !important;
  font-size: 13px !important;
}
/* Error message override */
.html-container div[style*="color: red"] {
  background: var(--red-soft) !important;
  border: 1px solid rgba(255,77,109,0.2) !important;
  border-radius: 8px !important;
  padding: 10px 14px !important;
  color: var(--red) !important;
  font-weight: 500 !important;
}
/* Success message override */
.html-container div[style*="color:green"],
.html-container div[style*="color: green"] {
  background: var(--accent-xglow) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 8px !important;
  padding: 10px 14px !important;
  color: var(--accent) !important;
  font-weight: 500 !important;
}

/* ── Info / description text ───────────────────────────────────── */
.info, .description, span.info {
  color: var(--text-3) !important;
  font-size: 11px !important;
}

/* ── Number input arrows ───────────────────────────────────────── */
.number button {
  background: var(--bg-raised) !important;
  border-color: var(--border) !important;
  color: var(--text-2) !important;
  padding: 4px 8px !important;
}
.number button:hover { color: var(--accent) !important; }

/* ── Accordion / Collapsible ───────────────────────────────────── */
.accordion, details {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}
.accordion-header, summary {
  color: var(--text-2) !important;
  font-size: 13px !important; font-weight: 600 !important;
  cursor: pointer !important;
}

/* ── Loader / spinner ──────────────────────────────────────────── */
.loader, .loading {
  border-color: var(--bg-raised) var(--accent) var(--bg-raised) var(--bg-raised) !important;
}

/* ── Row spacing ───────────────────────────────────────────────── */
.row { gap: 12px !important; }

/* ── Generate button row ────────────────────────────────────────── */
.btn-row {
  display: flex !important; gap: 10px !important;
  align-items: stretch !important;
}
"""


def build_app(inference_fct: Callable, engine=None, theme: str = "dark") -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Fish Speech") as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <div class="app-header-icon">🐟</div>
          <div>
            <div class="app-header-title">Fish Speech</div>
            <div class="app-header-sub">Neural Text-to-Speech Studio</div>
          </div>
          <div class="app-header-badge">S2-PRO</div>
        </div>
        """)

        # Force dark theme via URL param
        app.load(
            None, None,
            js="() => { const p = new URLSearchParams(window.location.search); if (!p.has('__theme')) { p.set('__theme', 'dark'); window.location.search = p.toString(); } }",
        )

        # ── Main layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left column: Input + Settings ──────────────────────────────
            with gr.Column(scale=5):

                # Input text
                text = gr.Textbox(
                    label=i18n("Input Text"),
                    placeholder=TEXTBOX_PLACEHOLDER,
                    lines=12,
                    show_copy_button=True,
                )

                # Settings tabs
                with gr.Tabs():
                    with gr.Tab(label="⚙  Generation"):
                        with gr.Row():
                            chunk_length = gr.Slider(
                                label=i18n("Chunk Length"),
                                info="Tokens per iterative chunk. 0 = off.",
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

                    with gr.Tab(label="🎙  Reference Voice"):
                        gr.Markdown(
                            "Upload 5–10 s of clean speech to clone the voice. "
                            "Or pick a saved voice from the dropdown."
                        )
                        with gr.Row():
                            use_memory_cache = gr.Radio(
                                label=i18n("Memory Cache"),
                                choices=["on", "off"],
                                value="on",
                            )
                        with gr.Row():
                            reference_id = gr.Dropdown(
                                label=i18n("Saved Voice"),
                                choices=[""] + (engine.list_reference_ids() if engine else []),
                                value="",
                                info="Leave empty to use uploaded audio",
                                allow_custom_value=True,
                            )
                            refresh_btn = gr.Button("↻", scale=0, min_width=44, variant="secondary")

                        reference_audio = gr.Audio(
                            label=i18n("Reference Audio"),
                            type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label=i18n("Reference Transcript"),
                            lines=2,
                            placeholder="Type the exact words spoken in the reference audio…",
                            value="",
                        )

                        gr.Markdown("---")
                        gr.Markdown("**Save as named voice**")
                        with gr.Row():
                            save_voice_name = gr.Textbox(
                                label=i18n("Voice Name"),
                                placeholder="e.g. narrator",
                                scale=3,
                            )
                            save_voice_btn = gr.Button(
                                i18n("Save Voice"), scale=1, variant="secondary"
                            )
                        save_voice_status = gr.HTML()

            # ── Right column: Output + Controls ────────────────────────────
            with gr.Column(scale=4):

                # Status: error + progress
                error = gr.HTML(visible=True)
                progress = gr.HTML(value="", visible=True)

                # Audio output
                audio = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy",
                    interactive=False,
                )

                # ZIP download (line-by-line)
                zip_file = gr.File(
                    label=i18n("Download ZIP  (Line-by-line mode)"),
                    interactive=False,
                )

                gr.HTML("<div style='height:4px'></div>")

                # Mode selector
                mode = gr.Radio(
                    label=i18n("Generation Mode"),
                    choices=["Normal", "Line-by-line"],
                    value="Normal",
                    info="Line-by-line: each paragraph → separate WAV file, packed into a ZIP",
                )

                gr.HTML("<div style='height:4px'></div>")

                # Action buttons
                with gr.Row(elem_classes=["btn-row"]):
                    generate = gr.Button(
                        value="▶  " + i18n("Generate"),
                        variant="primary",
                        scale=3,
                    )
                    stop_btn = gr.Button(
                        value="⏹  " + i18n("Stop"),
                        variant="stop",
                        scale=1,
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
        )

        stop_btn.click(fn=cancel_generation, cancels=[generate_event])

        def refresh_voices():
            ids = engine.list_reference_ids() if engine else []
            return gr.Dropdown(choices=[""] + ids)

        refresh_btn.click(refresh_voices, outputs=reference_id)

        def save_voice(audio_path, ref_text, voice_name):
            if not voice_name or not voice_name.strip():
                return "<div style='color:red'>Please enter a voice name.</div>"
            if not audio_path:
                return "<div style='color:red'>Please upload a reference audio file first.</div>"
            if not ref_text or not ref_text.strip():
                return "<div style='color:red'>Please enter the reference text first.</div>"
            if engine is None:
                return "<div style='color:red'>Engine not available.</div>"
            try:
                engine.add_reference(voice_name.strip(), audio_path, ref_text.strip())
                return f"<div style='color:green'>Voice '<b>{voice_name.strip()}</b>' saved.</div>"
            except FileExistsError:
                return f"<div style='color:red'>Name '<b>{voice_name.strip()}</b>' already exists.</div>"
            except Exception as e:
                return f"<div style='color:red'>Error: {e}</div>"

        save_voice_btn.click(
            save_voice,
            inputs=[reference_audio, reference_text, save_voice_name],
            outputs=[save_voice_status],
        )

    return app
