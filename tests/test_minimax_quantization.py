"""Optional Apple Silicon checks with small real MLX weights, no model downloads.

Run with the MiniMax environment's Python; the standard-library service suite skips them.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import minimax_generate


@unittest.skipUnless(importlib.util.find_spec("mlx"), "requires the optional Apple Silicon MLX environment")
class QuantizedEncoderTest(unittest.TestCase):
    def test_image_rows_replace_only_image_tokens_between_text_and_boundaries(self):
        import numpy as np
        from mlx_vlm.models.qwen3_vl.language import LanguageModel

        mx = self.mx
        ids = mx.array([[10, 99, 99, 11, 12, 99, 99, 13, 14]])
        text = mx.arange(36).reshape(1, 9, 4).astype(mx.float32)
        visual = mx.arange(16).reshape(4, 4).astype(mx.float32) + 100
        tags = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0])
        deepstack = [visual + 200]
        def hidden(input_ids, positions, **kwargs):
            self.assertTrue(mx.array_equal(kwargs["visual_pos_masks"], ids == 99).item())
            self.assertIs(kwargs["deepstack_visual_embeds"], deepstack)
            return kwargs["inputs_embeds"]
        encoder = SimpleNamespace(
            dtype=mx.float32, image_token_id=99,
            build_request=lambda prompt, images: (ids, tags, (np.zeros((16, 4)), np.array([[1, 2, 4], [1, 2, 4]]))),
            vision=lambda *args, **kwargs: (visual, deepstack),
            language=SimpleNamespace(embed_tokens=lambda tokens: text), _hidden_states=hidden)
        with patch.object(LanguageModel, "get_rope_index", return_value=(None, None)):
            result, result_tags = minimax_generate.encode_images(encoder, "move", [object(), object()])
        expected = np.asarray(text).copy()
        expected[0, [1, 2, 5, 6]] = np.asarray(visual)
        np.testing.assert_array_equal(np.asarray(result), expected)
        np.testing.assert_array_equal(result_tags, tags)
        encoder.vision = lambda *args, **kwargs: (visual[:3], deepstack)
        with self.assertRaisesRegex(ValueError, "do not match 4 image tokens"):
            minimax_generate.encode_images(encoder, "move", [object(), object()])

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
