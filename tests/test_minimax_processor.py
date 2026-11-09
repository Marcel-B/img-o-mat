"""H3 still images must not initialize Transformers' torch-based video processor."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import minimax_generate
import video


class ProcessorTest(unittest.TestCase):
    def test_only_local_image_processor_is_loaded(self):
        loader = Mock(return_value=object())
        # No AutoProcessor or video processor is exposed by this fake module.
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(
                Qwen2VLImageProcessorPil=SimpleNamespace(from_pretrained=loader))}):
            processor = minimax_generate.image_processor(Path("/snapshot/FL2VA/text_encoder"))
        loader.assert_called_once_with("/snapshot/FL2VA/processor", local_files_only=True)
        self.assertIs(processor.image_processor, loader.return_value)

    def test_original_phase_error_survives_subprocess_wrapper(self):
        self.assertEqual(video.runner_error([
            "ImportError:", "Qwen3VLVideoProcessor requires PyTorch >= 2.6",
            "Traceback (most recent call last):", "subprocess.CalledProcessError: encode failed",
        ]), "ImportError: Qwen3VLVideoProcessor requires PyTorch >= 2.6")
        self.assertEqual(video.runner_error(["ValueError: wrong shape"]), "ValueError: wrong shape")
        self.assertIsNone(video.runner_error(["loading weights"]))
