import os
from argparse import ArgumentParser
from pathlib import Path


import gradio as gr
import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.webui import CUSTOM_CSS, DARK_THEME, build_app
from tools.webui.inference import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fp8", action="store_true",
                        help="Quantize LLM weights to float8 via torchao (for 12-20 GB VRAM)")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


def _patch_init_model_for_fp8():
    """Monkey-patch init_model to load on CPU → quantize to FP8 → move to CUDA.

    This avoids the ~20 GB VRAM peak that happens when loading BF16 weights
    directly to GPU. With torchao float8_weight_only the model lands at ~10 GB.
    """
    import fish_speech.models.text2semantic.inference as _inf_module
    _orig_init_model = _inf_module.init_model

    def _fp8_init_model(checkpoint_path, device, precision, compile=False):
        from torchao.quantization import float8_weight_only, quantize_

        logger.info("FP8 mode: loading model to CPU first…")
        model = _orig_init_model(checkpoint_path, device="cpu", precision=precision, compile=False)

        logger.info("FP8 mode: quantizing Linear weights to float8…")
        quantize_(model, float8_weight_only())

        logger.info(f"FP8 mode: moving quantized model to {device}…")
        model = model.to(device=device)

        if compile:
            from fish_speech.models.text2semantic.inference import (
                decode_one_token_ar,
                decode_one_token_naive,
            )
            model.decode_one_token = torch.compile(
                decode_one_token_ar if hasattr(model, "fast_embeddings") else decode_one_token_naive,
                mode="default",
                fullgraph=True,
            )
        return model

    _inf_module.init_model = _fp8_init_model


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    if args.fp8:
        _patch_init_model_for_fp8()

    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        logger.info("XPU is available, running on XPU.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    logger.info("Warming up done, launching the web UI...")

    # Get the inference function with the immutable arguments
    inference_fct = get_inference_wrapper(inference_engine)

    app = build_app(inference_fct, inference_engine, args.theme)
    app.launch(share=True, theme=DARK_THEME, css=CUSTOM_CSS, server_name="0.0.0.0")
