"""MiniMax downloads must use a mounted T7, including all Hugging Face caches."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import common
import fetch
import probe


class ModelStorageTest(unittest.TestCase):
    def test_t7_must_be_a_mount_and_cache_must_resolve_inside_it(self):
        with patch.object(common.os.path, "ismount", return_value=False):
            self.assertIn("nicht angeschlossen", common.cache_volume_error("/Volumes/T7/models", "T7"))
        with patch.object(common.os.path, "ismount", return_value=True):
            self.assertIn("Modellordner muss", common.cache_volume_error("/Users/test/models", "T7"))
            self.assertIsNone(common.cache_volume_error("/Volumes/T7/models", "T7"))

    def test_cache_symlinks_cannot_redirect_downloads_internally(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical = os.path.realpath(directory)
            cache = Path(canonical) / "models"
            cache.mkdir()
            (cache / "hub").symlink_to("/Users/test/internal-hub")
            realpath = os.path.realpath
            def resolved(path):
                result = realpath(path)
                return result.replace(canonical, "/Volumes/T7", 1) if result.startswith(canonical) else result
            with patch.object(common.os.path, "realpath", side_effect=resolved), \
                 patch.object(common.os.path, "ismount", return_value=True):
                self.assertIn("außerhalb von T7", common.cache_volume_error(str(cache), "T7"))

    def test_worker_refuses_h3_before_creating_any_internal_cache(self):
        for repository in ("pipenetwork/minimax-h3-mlx-4bit", "MiniMaxAI/MiniMax-H3",
                           "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit"):
            with self.subTest(repository=repository), tempfile.TemporaryDirectory() as directory:
                asked = {"repos": [repository], "requiredVolume": None}
                with patch.object(fetch.common, "settings", return_value={"modelCache": directory}), \
                     patch.object(fetch.common, "prepare_environment") as prepare, \
                     patch.object(common.os.path, "ismount", return_value=False), \
                     patch.object(fetch.os, "makedirs") as mkdir, \
                     patch.object(fetch.sys, "stdin", io.StringIO(json.dumps(asked))), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(fetch.main(), 1)
                    self.assertIn("T7", output.getvalue())
                    mkdir.assert_not_called()
                    prepare.assert_not_called()

    def test_inherited_huggingface_caches_are_overridden(self):
        with patch.dict(os.environ, {"HF_HUB_CACHE": "/internal/hub", "HF_XET_CACHE": "/internal/xet"}):
            common.prepare_environment({"modelCache": "/Volumes/T7/models", "tokenFiles": []})
            for key, tail in (("HF_HUB_CACHE", "hub"), ("HUGGINGFACE_HUB_CACHE", "hub"),
                              ("HF_XET_CACHE", "xet"), ("HF_ASSETS_CACHE", "assets")):
                self.assertEqual(os.environ[key], "/Volumes/T7/models/" + tail)

    def test_new_t7_cache_uses_the_drive_free_space_before_folder_creation(self):
        values = {**common.settings(), "modelCache": "/Volumes/T7/new-models"}
        with patch.object(probe.common, "settings", return_value=values), \
             patch.object(probe, "prune"), patch.object(probe, "memory_gb", return_value=32), \
             patch.object(probe.os.path, "exists", return_value=False), \
             patch.object(probe.os.path, "ismount", return_value=True), \
             patch.object(probe.shutil, "disk_usage", return_value=SimpleNamespace(free=100e9)) as usage, \
             patch.object(probe.sys, "stdin", io.StringIO('{"repos": []}')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(probe.main(), 0)
        usage.assert_called_once_with("/Volumes/T7")
        state = json.loads(output.getvalue())
        self.assertEqual(state["freeGb"], 100)
        self.assertTrue(state["modelCacheMounted"])
        self.assertIsNone(state["modelCacheStorageError"])

    def test_queued_download_rechecks_storage_before_each_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            download = Mock()
            errors = [None, None, "T7 wurde getrennt."]
            with patch.object(fetch.common, "settings", return_value={"modelCache": directory}), \
                 patch.object(fetch.common, "prepare_environment"), \
                 patch.object(fetch.common, "cache_volume_error", side_effect=errors) as check, \
                 patch.object(fetch.sys, "stdin", io.StringIO(json.dumps({"repos": ["a", "b"], "requiredVolume": "T7"}))), \
                 patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fetch.main(), 1)
                self.assertEqual(check.call_count, 3)
                download.assert_called_once_with("a", cache_dir=os.path.join(directory, "hub"),
                                                 revision=None, allow_patterns=None)


if __name__ == "__main__":
    unittest.main()
