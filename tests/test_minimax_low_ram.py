"""Sequential H3 phases: fail fast, inherit cancellation's process group, clean normal failures."""

import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import minimax_low_ram


class LowRamTest(unittest.TestCase):
    def arguments(self, folder, extra=None):
        return ["--output", str(Path(folder) / "clip.mp4"), "--checkpoint", "/base", "--transformer", "/quant",
                *(extra or []), "--", "a fox"]

    def test_phases_finish_sequentially_before_muxing(self):
        stages = []
        phase_folder = None

        def run(args, **kwargs):
            nonlocal phase_folder
            self.assertTrue(kwargs["check"])
            self.assertNotIn("start_new_session", kwargs)  # launch.py's group cancels every phase.
            stage, path = args[-2:]
            phase_folder = Path(path).parent
            stages.append(stage)
            if stage == "sample":
                (phase_folder / "result.json").write_text(json.dumps({"fps": 24, "sampleRate": 32000}))

        save = Mock()
        fake_numpy = SimpleNamespace(load=lambda path, **kwargs: str(path))
        with tempfile.TemporaryDirectory() as folder, patch.object(minimax_low_ram.subprocess, "run", side_effect=run), \
             patch.dict(sys.modules, {"numpy": fake_numpy, "minimax_h3_mlx.media": SimpleNamespace(save_mp4=save)}), \
             contextlib.redirect_stdout(io.StringIO()):
            minimax_low_ram.generate(Path("/source"), Path("/encoder"), self.arguments(folder, ["--image", "/image", "--anchor", "last"]))
            self.assertEqual(stages, ["encode", "keyframes", "sample", "video", "audio"])
            self.assertEqual(save.call_count, 1)
            self.assertFalse(phase_folder.exists())

    def test_failure_stops_before_loading_another_component(self):
        phases = []
        def fail(args, **kwargs):
            phases.append(Path(args[-1]).parent)
            raise subprocess.CalledProcessError(1, args)

        with tempfile.TemporaryDirectory() as folder, patch.object(minimax_low_ram.subprocess, "run", side_effect=fail), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            minimax_low_ram.generate(Path("/source"), Path("/encoder"), self.arguments(folder))
        self.assertEqual(len(phases), 1)
        self.assertFalse(phases[0].exists())

    def test_direct_cli_also_rejects_large_32gb_requests(self):
        for extra in (["--duration", "10"], ["--width", "1344"], ["--image", "/image"]):
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(minimax_low_ram.subprocess, "run") as run, self.assertRaises(SystemExit):
                minimax_low_ram.generate(Path("/source"), Path("/encoder"), self.arguments(folder, extra))
            run.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IMG_O_MAT_MINIMAX_SOURCE"),
                     "requires MLX and IMG_O_MAT_MINIMAX_SOURCE pointing to the pinned runner")
class SamplingParityTest(unittest.TestCase):
    def test_decoder_phases_load_only_their_own_weights(self):
        sys.path.insert(0, os.environ["IMG_O_MAT_MINIMAX_SOURCE"])
        import mlx.core as mx
        import numpy as np
        from minimax_h3_mlx import load
        from minimax_h3_mlx.config import PipelineConfig

        dit = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))
        video = SimpleNamespace(config=SimpleNamespace(latent_channels=24, latents_mean=[0] * 24, latents_std=[1] * 24),
                                decode=lambda latents: mx.zeros((1, 3, 5, 32, 32)))
        audio = SimpleNamespace(config=SimpleNamespace(latent_channels=32, latents_mean=[0] * 32, latents_std=[1] * 32),
                                decode=lambda latents: mx.zeros((2, 1, 10)))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            mx.save_safetensors(str(folder / "video-latents.safetensors"), {"rows": mx.ones((2, 96))})
            (folder / "video-shape.json").write_text(json.dumps([2, 2, 2]))
            mx.save_safetensors(str(folder / "audio-latents.safetensors"), {"rows": mx.ones((4, 32))})
            (folder / "audio-shape.json").write_text("2")
            for stage in ("video", "audio"):
                with self.subTest(stage=stage), patch.object(load, "load_dit") as load_dit, \
                     patch.object(load, "load_video_vae", return_value=video) as load_video, \
                     patch.object(load, "load_audio_vae", return_value=audio) as load_audio, \
                     patch.object(minimax_low_ram, "pipeline_parts", return_value=(dit, video, audio, PipelineConfig())), \
                     contextlib.redirect_stdout(io.StringIO()):
                    minimax_low_ram.run_stage(stage, {"checkpoint": "/base"}, folder)
                    load_dit.assert_not_called()
                    self.assertEqual(load_video.call_count, int(stage == "video"))
                    self.assertEqual(load_audio.call_count, int(stage == "audio"))
            self.assertEqual(np.load(folder / "video.npy").shape, (5, 32, 32, 3))
            self.assertEqual(np.load(folder / "audio.npy").shape, (2, 10))

    def test_phase_boundary_preserves_the_upstream_sampler_and_seed(self):
        sys.path.insert(0, os.environ["IMG_O_MAT_MINIMAX_SOURCE"])
        import mlx.core as mx
        import numpy as np
        from minimax_h3_mlx import load
        from minimax_h3_mlx.config import PipelineConfig
        from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

        class TinyDit:
            config = SimpleNamespace(patch_size=(1, 2, 2))
            def __call__(self, video, audio, *args, **kwargs):
                return video * 0.25, audio * 0.125

        class Encoder:
            def encode(self, *args):
                return mx.ones((1, 3, 64), dtype=mx.bfloat16), np.ones(3, dtype=np.int64)

        class Capture(MiniMaxH3Pipeline):
            def _decode_video(self, rows, *shape):
                return np.array(rows)
            def _decode_audio(self, rows, *shape):
                return np.array(rows)
            def _encode_keyframes(self, *args):
                return mx.ones((2, 96))

        video = SimpleNamespace(config=SimpleNamespace(spatial_compression_ratio=16, latent_channels=24))
        audio = SimpleNamespace(config=SimpleNamespace(latent_channels=32, sampling_rate=32000))
        config = PipelineConfig()
        for anchors in ([], ["first", "last"]):
            with self.subTest(anchors=anchors), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                embeds, tags = Encoder().encode()
                mx.save_safetensors(str(folder / "conditioning.safetensors"), {"embeds": embeds})
                np.save(folder / "tags.npy", tags)
                mx.save_safetensors(str(folder / "keyframes.safetensors"), {"rows": mx.ones((2, 96))})
                request = {"checkpoint": "/base", "transformer": "/quant", "prompt": "fox", "duration": 5,
                           "seed": 17, "steps": 3, "height": 32, "width": 32,
                           "image": ["image"] * len(anchors), "anchor": anchors}
                with patch.object(MiniMaxH3Pipeline, "_ensure_cache"), \
                     patch.object(load, "load_dit", return_value=TinyDit()) as load_dit, \
                     patch.object(load, "load_video_vae") as load_video, \
                     patch.object(load, "load_audio_vae") as load_audio, \
                     patch.object(minimax_low_ram, "pipeline_parts", return_value=(TinyDit(), video, audio, config)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    reference = Capture(TinyDit(), Encoder(), video, audio, config)(
                        "fox", duration_seconds=5, seed=17, num_inference_steps=3, height=32, width=32,
                        images=request["image"] or None, keyframe_anchors=tuple(anchors),
                    )
                    minimax_low_ram.run_stage("sample", request, folder)
                    load_dit.assert_called_once()
                    load_video.assert_not_called()
                    load_audio.assert_not_called()
                actual_video = mx.load(str(folder / "video-latents.safetensors"))["rows"]
                actual_audio = mx.load(str(folder / "audio-latents.safetensors"))["rows"]
                np.testing.assert_array_equal(np.array(actual_video), reference.video)
                np.testing.assert_array_equal(np.array(actual_audio), reference.audio)


if __name__ == "__main__":
    unittest.main()
