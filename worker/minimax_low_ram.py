"""H3's conditioning, sampling and decoding in sequential, short-lived processes.

Only small conditioning/latent tensors cross phase boundaries. A process exit releases
MLX allocations and mapped checkpoint weights before the next component is constructed.
The pinned upstream pipeline still supplies the sampler, keyframe RNG and VAE decoding.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def read_json(path):
    with open(path) as file:
        return json.load(file)


def images(request):
    from PIL import Image, ImageOps

    result = []
    for path in request["image"]:
        with Image.open(path) as image:
            result.append(ImageOps.exif_transpose(image).convert("RGB"))
    return result or None


def pipeline_parts(request):
    from minimax_h3_mlx.config import DiTConfig, PipelineConfig

    root = Path(request["checkpoint"])
    video = read_json(root / "video_vae" / "config.json")
    source = read_json(root / "video_vae" / "source" / "config.json")
    audio = read_json(root / "audio_vae" / "config.json")
    return (
        SimpleNamespace(config=DiTConfig.from_json(Path(request["transformer"]) / "config.json")),
        SimpleNamespace(config=SimpleNamespace(
            spatial_compression_ratio=math.prod(source["space_down"]), latent_channels=source["z_channels"],
            latents_mean=video["latents_mean"], latents_std=video["latents_std"],
        )),
        SimpleNamespace(config=SimpleNamespace(
            latent_channels=audio["latent_channels"], sampling_rate=audio["sample_rate"],
            latents_mean=audio["latents_mean"], latents_std=audio["latents_std"],
        )),
        PipelineConfig.from_model_index(root / "model_index.json"),
    )


def run_stage(stage: str, request: dict, folder: Path) -> None:
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx import load
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    # Local per-process controls; never raise macOS's global wired-memory limit.
    mx.set_cache_limit(256 << 20)
    mx.set_memory_limit(24 << 30)
    root = Path(request["checkpoint"])
    if stage == "encode":
        from minimax_generate import install_encoder

        encoder_class = install_encoder(Path(request["encoder"]))
        encoder = encoder_class(root / "text_encoder", load_vision=bool(request["image"]))
        embeds, tags = encoder.encode(request["prompt"], images(request))
        mx.save_safetensors(str(folder / "conditioning.safetensors"), {"embeds": embeds})
        np.save(folder / "tags.npy", tags)
    else:
        dit, video, audio, config = pipeline_parts(request)
        pipe = MiniMaxH3Pipeline(dit, None, video, audio, config)
        if stage == "keyframes":
            pipe.video_vae = load.load_video_vae(root / "video_vae")
            rows = pipe._encode_keyframes(images(request), request["height"], request["width"])
            mx.save_safetensors(str(folder / "keyframes.safetensors"), {"rows": rows})
        elif stage == "sample":
            class CachedEncoder:
                def encode(self, prompt, given_images):
                    return mx.load(str(folder / "conditioning.safetensors"))["embeds"], np.load(folder / "tags.npy")

            class SamplingPipeline(MiniMaxH3Pipeline):
                def _encode_keyframes(self, given_images, height, width):
                    return mx.load(str(folder / "keyframes.safetensors"))["rows"]

                def _decode_video(self, rows, num_frames, height, width):
                    mx.save_safetensors(str(folder / "video-latents.safetensors"), {"rows": rows})
                    (folder / "video-shape.json").write_text(json.dumps([num_frames, height, width]))
                    return np.empty((0, 1, 1, 3), dtype=np.uint8)

                def _decode_audio(self, rows, num_frames):
                    mx.save_safetensors(str(folder / "audio-latents.safetensors"), {"rows": rows})
                    (folder / "audio-shape.json").write_text(json.dumps(num_frames))
                    return np.empty((2, 0), dtype=np.float32)

            from minimax_h3_mlx.streaming import load_streaming_dit

            # Only static weights and one reusable block slot are resident; AdaLN is streamed too.
            model, provider = load_streaming_dit(request["transformer"], stream_block_group_size=1, verbose=True)
            pipe = SamplingPipeline(model, CachedEncoder(), video, audio, config)
            pipe._block_provider = provider
            result = pipe(
                request["prompt"], duration_seconds=request["duration"],
                num_inference_steps=request["steps"], seed=request["seed"],
                images=[True] * len(request["image"]) or None, keyframe_anchors=tuple(request["anchor"]),
                height=request["height"], width=request["width"],
            )
            (folder / "result.json").write_text(json.dumps({"fps": result.fps, "sampleRate": result.sample_rate}))
        elif stage == "video":
            pipe.video_vae = load.load_video_vae(root / "video_vae")
            rows = mx.load(str(folder / "video-latents.safetensors"))["rows"]
            frames = pipe._decode_video(rows, *read_json(folder / "video-shape.json"))
            np.save(folder / "video.npy", frames)
        elif stage == "audio":
            pipe.audio_vae = load.load_audio_vae(root / "audio_vae")
            rows = mx.load(str(folder / "audio-latents.safetensors"))["rows"]
            waveform = pipe._decode_audio(rows, read_json(folder / "audio-shape.json"))
            np.save(folder / "audio.npy", waveform)
        else:
            raise ValueError(f"Unknown H3 phase: {stage}")
    mx.synchronize()
    peak = mx.get_peak_memory() / (1 << 30)
    print(f"MINIMAX_PEAK_GB={peak:.3f}", flush=True)


def generate(source: Path, encoder: Path, remaining: list[str], *, streaming_source: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3: sequential 32 GB mode")
    parser.add_argument("prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--anchor", action="append", default=[], choices=["first", "last"])
    args = parser.parse_args(remaining)
    if (args.width, args.height, args.duration) != (768, 768, 5):
        parser.error("Der 32-GB-Modus ist vorerst auf 768 × 768 und 5 Sekunden begrenzt.")
    if len(args.image) != len(args.anchor):
        parser.error("Jedes Bild braucht einen first/last-Anker.")
    if streaming_source is None or not (streaming_source / "minimax_h3_mlx" / "streaming.py").is_file():
        parser.error("MiniMax-Blockstreaming fehlt; install-worker.sh auf dem Mac ausführen.")
    request = {**vars(args), "source": str(source), "encoder": str(encoder), "streamingSource": str(streaming_source)}
    # Keep phase artifacts alongside worker/video.py's temporary clip.
    with tempfile.TemporaryDirectory(prefix="phases-", dir=str(Path(args.output).parent)) as temporary:
        folder = Path(temporary)
        path = folder / "request.json"
        path.write_text(json.dumps(request))
        stages = ["encode", *(["keyframes"] if args.image else []), "sample", "video", "audio"]
        for stage in stages:
            fraction = {"encode": 0.01, "keyframes": 0.03, "sample": 0.05, "video": 0.96, "audio": 0.97}[stage]
            print("IMGOMAT " + json.dumps({"event": "stage", "stage": "saving" if stage in ("video", "audio") else "loading",
                                           "fraction": fraction}), flush=True)
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", stage, str(path)], check=True)
        import numpy as np
        from minimax_h3_mlx.media import save_mp4

        result = read_json(folder / "result.json")
        save_mp4(args.output, np.load(folder / "video.npy", mmap_mode="r"), result["fps"],
                 np.load(folder / "audio.npy"), result["sampleRate"])


if __name__ == "__main__":
    stage, path = sys.argv[2:]
    request = read_json(path)
    # Separate processes prevent the identically named fork packages from mixing in sys.modules.
    sys.path.insert(0, request["streamingSource"] if stage == "sample" else request["source"])
    run_stage(stage, request, Path(path).parent)
