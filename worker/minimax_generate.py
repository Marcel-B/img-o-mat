"""Run the pinned H3 CLI with its own Qwen conditioner quantized to 4-bit.

The MLX-Serve pack contains the same 50 conditioning layers in MLX affine format.
Keep PipeNetwork's prompt construction and pre-norm forward; replace only weight loading.
No model files are fetched here. Original config/tokenizer/processor come from the local
FL2VA snapshot, quantized language and vision weights from the separate encoder pack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def image_processor(model_dir: Path):
    from transformers import Qwen2VLImageProcessorPil

    # H3 only uses still images. AutoProcessor also constructs a video processor,
    # which unnecessarily requires torch/torchvision in an otherwise MLX runner.
    return SimpleNamespace(image_processor=Qwen2VLImageProcessorPil.from_pretrained(
        str(model_dir.parent / "processor"), local_files_only=True))


def encode_images(encoder, prompt: str, images):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.qwen3_vl.language import LanguageModel

    input_ids, tags, (pixels, grid) = encoder.build_request(prompt, images)
    if encoder.vision is None:
        raise ValueError("H3 image conditioning requires the vision encoder")
    grid = mx.array(grid.astype(np.int32))
    features, deepstack = encoder.vision(mx.array(pixels).astype(encoder.dtype), grid, output_hidden_states=True)
    embeds = encoder.language.embed_tokens(input_ids)
    image_mask = input_ids == encoder.image_token_id
    # The vision output has only image rows; text and image-boundary tokens stay intact.
    positions = np.flatnonzero(np.asarray(image_mask).reshape(-1))
    if features.shape != (len(positions), embeds.shape[-1]):
        raise ValueError(f"H3 image features {features.shape} do not match {len(positions)} image tokens")
    flattened = embeds.reshape(-1, embeds.shape[-1])
    flattened[mx.array(positions, dtype=mx.uint32)] = features.astype(embeds.dtype)
    embeds = flattened.reshape(embeds.shape)
    position_ids, _ = LanguageModel.get_rope_index(
        encoder, input_ids, image_grid_thw=grid, video_grid_thw=None, attention_mask=None)
    hidden = encoder._hidden_states(input_ids, position_ids, inputs_embeds=embeds,
                                   visual_pos_masks=image_mask, deepstack_visual_embeds=deepstack)
    mx.eval(hidden)
    return hidden, tags


def encoder_weights(path: Path, load_vision: bool) -> dict:
    import mlx.core as mx

    weights = mx.load(str(path))
    selected = {}
    for key, tensor in weights.items():
        if key.startswith("model."):
            selected["language." + key[len("model."):]] = tensor
        elif load_vision and key.startswith("visual."):
            selected["vision." + key[len("visual."):]] = tensor
    return selected


def vision_weights(weights: dict) -> dict:
    # This pack always stores PyTorch's (out, in, time, height, width) kernels.
    # Avoid a shape heuristic: small square kernels can look valid in both layouts.
    key = "patch_embed.proj.weight"
    return {**weights, key: weights[key].transpose(0, 2, 3, 4, 1)}


def load_quantized(module, weights: dict, dtype, *, allow_missing: tuple[str, ...] = ()) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    # Convert only matrices that actually have quantization scales in the checkpoint.
    # In particular, token and position embeddings stay dense in this pack.
    nn.quantize(module, group_size=64, bits=4,
                class_predicate=lambda name, layer: name + ".scales" in weights)
    expected = dict(tree_flatten(module.parameters()))
    missing = set(expected) - weights.keys() - set(allow_missing)
    unexpected = weights.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"H3 encoder weights mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")
    for name, tensor in weights.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f"H3 encoder shape mismatch for {name}: {tensor.shape} != {expected[name].shape}")
        # Casting packed uint32 matrices to bf16 silently destroys their quantized values.
        if name.endswith(".weight") and name[:-len("weight")] + "scales" in weights:
            if tensor.dtype != mx.uint32:
                raise ValueError(f"H3 quantized weight {name} must be uint32")
        elif tensor.dtype != mx.uint32:
            tensor = tensor.astype(dtype)
        weights[name] = tensor
    module.load_weights(list(weights.items()), strict=False)
    mx.eval(module.parameters())


def install_encoder(encoder_dir: Path):
    import minimax_h3_mlx.text_encoder as upstream

    with open(encoder_dir / "config.json") as file:
        config = json.load(file)
    if config.get("quantization") != {"group_size": 64, "bits": 4, "mode": "affine"}:
        raise ValueError("H3 encoder requires MLX affine 4-bit weights with group size 64")
    if config.get("text_encoder") != {
        "hidden": 5120, "layers": 50, "heads": 64, "kv_heads": 8,
        "head_dim": 128, "intermediate": 25600, "theta": 5000000.0,
    }:
        raise ValueError("H3 encoder geometry differs from the 50-layer Qwen3-VL-32B conditioner")

    class QuantizedEncoder(upstream.MiniMaxH3TextEncoder):
        def encode(self, prompt, images=None):
            if images:
                return encode_images(self, prompt, images)
            return super().encode(prompt, images)

        @property
        def processor(self):
            if self._processor is None:
                self._processor = image_processor(self._model_dir)
            return self._processor

        def _load_weights(self, model_dir, dtype, verbose):
            weights = encoder_weights(encoder_dir / "text_encoder.safetensors", self.vision is not None)
            for name, module in (("language", self.language), ("vision", self.vision)):
                if module is None:
                    continue
                prefix = name + "."
                part = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
                if name == "vision":
                    # The pack retains PyTorch's Conv3d layout; MLX needs channels last.
                    part = vision_weights(part)
                # The H3 repack omits final norm, which the inherited forward never applies.
                load_quantized(module, part, dtype, allow_missing=("norm.weight",) if name == "language" else ())
            self.skipped_tensors = 0

    upstream.MiniMaxH3TextEncoder = QuantizedEncoder
    return QuantizedEncoder


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True)
    parser.add_argument("--quantized-text-encoder", required=True)
    parser.add_argument("--low-ram", action="store_true")
    parser.add_argument("--streaming-source", type=Path)
    arguments, remaining = parser.parse_known_args()
    source = Path(arguments.source)
    sys.path.insert(0, str(source))
    if arguments.low_ram:
        from minimax_low_ram import generate
        generate(source, Path(arguments.quantized_text_encoder), remaining, streaming_source=arguments.streaming_source)
        return
    install_encoder(Path(arguments.quantized_text_encoder))
    script = source / "scripts" / "generate.py"
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
