from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


def build_app(inference_fct: Callable, engine=None, theme: str = "light") -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=100,
                                    maximum=400,
                                    value=300,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.85,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.5,
                                    value=1.05,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "5 to 10 seconds of reference audio, useful for specifying speaker."
                                    )
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["on", "off"],
                                    value="on",
                                )

                            with gr.Row():
                                reference_id = gr.Dropdown(
                                    label=i18n("Saved Voice"),
                                    choices=[""] + (engine.list_reference_ids() if engine else []),
                                    value="",
                                    info="Select a saved voice, or leave empty to use uploaded audio below",
                                    allow_custom_value=True,
                                )
                                refresh_btn = gr.Button("↻", scale=0, min_width=48)

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )
                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

                            gr.Markdown("---")
                            gr.Markdown("**Save uploaded audio as a named voice:**")
                            with gr.Row():
                                save_voice_name = gr.Textbox(
                                    label=i18n("Voice Name"),
                                    placeholder="e.g. my-voice",
                                    scale=2,
                                )
                                save_voice_btn = gr.Button(i18n("Save Voice"), scale=1)
                            save_voice_status = gr.HTML(visible=True)

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    progress = gr.HTML(value="", visible=True)
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    mode = gr.Radio(
                        label=i18n("Generation Mode"),
                        choices=["Normal", "Line-by-line"],
                        value="Normal",
                        info="Line-by-line: each non-empty line → separate numbered MP3",
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # Submit
        generate.click(
            inference_fct,
            [
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
            ],
            [audio, error, progress],
            concurrency_limit=1,
        )

        # Refresh voice dropdown
        def refresh_voices():
            ids = engine.list_reference_ids() if engine else []
            return gr.Dropdown(choices=[""] + ids)

        refresh_btn.click(refresh_voices, outputs=reference_id)

        # Save voice
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
                return f"<div style='color:green'>Voice '<b>{voice_name.strip()}</b>' saved successfully.</div>"
            except FileExistsError:
                return f"<div style='color:red'>Voice name '<b>{voice_name.strip()}</b>' already exists. Choose a different name.</div>"
            except Exception as e:
                return f"<div style='color:red'>Error: {e}</div>"

        save_voice_btn.click(
            save_voice,
            [reference_audio, reference_text, save_voice_name],
            [save_voice_status],
        )

    return app
