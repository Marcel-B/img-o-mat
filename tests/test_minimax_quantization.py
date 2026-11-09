"""Optional Apple Silicon checks with small real MLX weights, no model downloads.

Run with the MiniMax environment's Python; the standard-library service suite skips them.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import minimax_generate


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires the optional Apple Silicon MLX environment")
class QuantizedEncoderTest(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        self.mx, self.nn, self.flatten = mx, nn, tree_flatten

    def test_roundtrip_preserves_packed_weights_and_forward_result(self):
        mx, nn = self.mx, self.nn
        reference = nn.Sequential(nn.Embedding(8, 64), nn.Linear(64, 64))
        nn.quantize(reference, group_size=64, bits=4, class_predicate=lambda name, layer: name == "layers.1")
        weights = dict(self.flatten(reference.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text_encoder.safetensors"
            mx.save_safetensors(str(path), {"model." + key: value for key, value in weights.items()})
            loaded = minimax_generate.encoder_weights(path, load_vision=False)
        actual = nn.Sequential(nn.Embedding(8, 64), nn.Linear(64, 64))
        minimax_generate.load_quantized(actual, {key[len("language."):]: value for key, value in loaded.items()}, mx.float32)
        self.assertEqual(actual.layers[1].weight.dtype, mx.uint32)
        self.assertIsInstance(actual.layers[0], nn.Embedding)
        x = mx.array([[1, 2, 3]])
        self.assertTrue(mx.array_equal(actual(x), reference(x)).item())

    def test_missing_or_wrongly_shaped_weights_fail(self):
        mx, nn = self.mx, self.nn
        with self.assertRaisesRegex(ValueError, "missing"):
            minimax_generate.load_quantized(nn.Sequential(nn.Linear(64, 64)), {}, mx.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            minimax_generate.load_quantized(nn.Sequential(nn.Linear(64, 64, bias=False)),
                                           {"layers.0.weight": mx.zeros((1, 1))}, mx.float32)

    def test_text_only_loading_drops_vision_weights(self):
        mx = self.mx
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text_encoder.safetensors"
            mx.save_safetensors(str(path), {"model.embed_tokens.weight": mx.ones((8, 64)),
                                           "visual.pos_embed.weight": mx.ones((4, 64))})
            self.assertEqual(set(minimax_generate.encoder_weights(path, False)), {"language.embed_tokens.weight"})
            self.assertIn("vision.pos_embed.weight", minimax_generate.encoder_weights(path, True))

    def test_real_qwen_vision_layout_roundtrip(self):
        from mlx_vlm.models.qwen3_vl.config import VisionConfig
        from mlx_vlm.models.qwen3_vl.vision import VisionModel

        mx, nn = self.mx, self.nn
        config = VisionConfig(depth=1, hidden_size=64, intermediate_size=128, out_hidden_size=64,
                              num_heads=1, patch_size=2, temporal_patch_size=2,
                              num_position_embeddings=16, deepstack_visual_indexes=[])
        reference = VisionModel(config)
        nn.quantize(reference, group_size=64, bits=4,
                    class_predicate=lambda name, layer: isinstance(layer, nn.Linear))
        weights = dict(self.flatten(reference.parameters()))
        # The published pack keeps Conv3d kernels in PyTorch's channel-first layout.
        weights["patch_embed.proj.weight"] = weights["patch_embed.proj.weight"].transpose(0, 4, 1, 2, 3)
        actual = VisionModel(config)
        minimax_generate.load_quantized(actual, minimax_generate.vision_weights(weights), mx.float32)
        expected = dict(self.flatten(reference.parameters()))
        for key, value in self.flatten(actual.parameters()):
            self.assertTrue(mx.array_equal(value, expected[key]).item(), key)


if __name__ == "__main__":
    unittest.main()
