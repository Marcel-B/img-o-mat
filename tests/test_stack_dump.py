"""A hanging picture leaves the service running; SIGUSR1 must show where every thread stands without a restart."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StackDumpTest(unittest.TestCase):
    def test_usr1_writes_all_threads_to_stderr(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "worker.json"), "w") as out:
                json.dump({"python": sys.executable, "scripts": os.path.join(ROOT, "tests", "fake")}, out)
            config = os.path.join(root, "config.json")
            with open(config, "w") as out:
                json.dump({"root": root, "listen": "127.0.0.1", "port": 0, "minFreeGb": 0}, out)
            process = subprocess.Popen([sys.executable, os.path.join(ROOT, "service", "server.py")],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env={**os.environ, "IMG_O_MAT_CONFIG": config,
                                            "IMG_O_MAT_SOURCE_WORKER": os.path.join(ROOT, "worker")})
            try:
                # The first line says the server listens; the handler is registered before it.
                process.stdout.readline()
                process.send_signal(signal.SIGUSR1)
                time.sleep(1)
            finally:
                process.terminate()
                _, err = process.communicate(timeout=10)
            self.assertIn(b"Thread 0x", err)
            self.assertIn(b"serve_forever", err)


if __name__ == "__main__":
    unittest.main()
