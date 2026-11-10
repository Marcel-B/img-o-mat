"""Internal video scratch space is disposable on completion, error and cancellation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

WORKER = Path(__file__).resolve().parents[1] / "worker"
sys.path.insert(0, str(WORKER))
import common


class VideoWorkTest(unittest.TestCase):
    def test_active_job_survives_probe_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(common.create_video_work(directory))
            (work / "latent.bin").write_bytes(b"active")
            common.cleanup_video_work(directory)
            self.assertEqual((work / "latent.bin").read_bytes(), b"active")

    def test_orphan_is_removed_without_touching_other_data_or_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = Path(common.create_video_work(directory))
            (work / "latent.bin").write_bytes(b"orphan")
            other = root / "kept-models"
            other.mkdir()
            (other / "weights.bin").write_bytes(b"keep")
            (root / "img-o-mat-video-link").symlink_to(other, target_is_directory=True)
            with patch.object(common.os, "kill", side_effect=ProcessLookupError):
                common.cleanup_video_work(directory)
            self.assertFalse(work.exists())
            self.assertEqual((other / "weights.bin").read_bytes(), b"keep")
            self.assertTrue((root / "img-o-mat-video-link").is_symlink())

    def test_unreadable_owner_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(common.create_video_work(directory))
            (work / "owner-pid").write_text("unknown")
            common.cleanup_video_work(directory)
            self.assertTrue(work.exists())

    def test_launcher_cleans_normal_completion_error_and_cancelled_worker(self):
        for mode in ("done", "error", "cancel"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                script = root / "scripts"
                script.mkdir()
                (script / "video.py").write_text(
                    "import json,os,sys,time\nfrom pathlib import Path\n"
                    "request=json.load(sys.stdin)\n"
                    "work=Path(os.environ['IMG_O_MAT_VIDEO_WORK'])\n"
                    "(work/'phases').mkdir()\n(work/'phases'/'latent.bin').write_bytes(b'partial')\n"
                    "print(str(work),flush=True)\n"
                    "if request['mode']=='cancel': time.sleep(30)\n"
                    "sys.exit(1 if request['mode']=='error' else 0)\n"
                )
                scratch = root / "internal-work"
                (root / "worker.json").write_text(json.dumps({"scripts": str(script), "ltxPython": sys.executable,
                                                              "videoWork": str(scratch)}))
                process = subprocess.Popen([sys.executable, str(WORKER / "launch.py"), "video"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env={**os.environ, "IMG_O_MAT_WORKER_ROOT": str(root)})
                try:
                    process.stdin.write((json.dumps({"mode": mode}) + "\n").encode())
                    process.stdin.flush()
                    work = Path(process.stdout.readline().decode().strip())
                    self.assertEqual(work.parent, scratch)
                    if mode == "cancel":
                        self.assertTrue((work / "phases" / "latent.bin").exists())
                        process.stdin.close()
                    code = process.wait(timeout=5)
                    self.assertEqual(code == 0, mode == "done")
                    self.assertFalse(work.exists())
                    self.assertEqual(list(scratch.iterdir()), [])
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    for stream in (process.stdin, process.stdout, process.stderr):
                        stream.close()


    def test_launcher_ends_when_the_script_leaves_a_helper_behind(self):
        # A helper that inherited stdout would keep the SSH connection, and with it the service, waiting.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "scripts"
            script.mkdir()
            (script / "paint.py").write_text(
                "import subprocess,sys\n"
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                "print('IMGOMAT {\"event\": \"stage\", \"stage\": \"checking\"}',flush=True)\n"
                "sys.exit(1)\n"
            )
            (root / "worker.json").write_text(json.dumps({"scripts": str(script), "python": sys.executable,
                                                          "runners": {"mflux": {"python": sys.executable}}}))
            process = subprocess.Popen([sys.executable, str(WORKER / "launch.py"), "paint", "mflux"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env={**os.environ, "IMG_O_MAT_WORKER_ROOT": str(root)})
            try:
                process.stdin.write(b"{}\n")
                process.stdin.flush()
                result = []
                reader = threading.Thread(target=lambda: result.append(process.stdout.read()), daemon=True)
                reader.start()
                reader.join(timeout=10)
                self.assertFalse(reader.is_alive(), "stdout must end when the script ends")
                self.assertIn(b"checking", result[0])
                self.assertEqual(process.wait(timeout=5), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()


if __name__ == "__main__":
    unittest.main()
