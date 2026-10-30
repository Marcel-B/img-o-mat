"""/metrics and the Loki shipper: the text format, what a picture counts against the fake runners (tests/fake), and the
log lines Loki gets from a small local stand-in.

    python3 -m unittest discover tests
"""

from __future__ import annotations

import ast
import glob
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))

import config as settings  # noqa: E402
import server  # noqa: E402
from loki import LokiShipper  # noqa: E402
from metrics import Counter  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "fake")
SAMPLE = re.compile(r'^([a-z_:][a-z0-9_:]*)(\{[a-z_]+="(?:[^"\\]|\\.)*"(?:,[a-z_]+="(?:[^"\\]|\\.)*")*\})? ([-+0-9.e]+|\+Inf|NaN)$')


class FakeLoki(BaseHTTPRequestHandler):
    pushed: list = []
    status = 204

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        FakeLoki.pushed.append((self.path, json.loads(body)))
        self.send_response(FakeLoki.status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def value(text: str, name: str, labels: str = "") -> float | None:
    prefix = f"{name}{{{labels}}} " if labels else f"{name} "
    line = next((l for l in text.splitlines() if l.startswith(prefix)), None)
    return None if line is None else float(line.rsplit(" ", 1)[1])


class MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="img-o-mat-metrics-")
        with open(os.path.join(cls.root, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "scripts": FAKE}, out)
        os.environ["IMG_O_MAT_SOURCE_WORKER"] = os.path.join(ROOT, "worker")
        cls.loki = HTTPServer(("127.0.0.1", 0), FakeLoki)
        threading.Thread(target=cls.loki.serve_forever, daemon=True).start()
        config = settings.Config({
            "root": cls.root, "listen": "127.0.0.1", "port": 0, "systemMemoryGb": 24, "minFreeGb": 0,
            "loki": {"url": f"http://127.0.0.1:{cls.loki.server_address[1]}/", "host": "probe", "intervalSeconds": 0.2},
        })

        class Handler(server.Handler):
            service = server.Service(config)

        cls.service = Handler.service
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.loki.shutdown()
        server.shipper = None
        shutil.rmtree(cls.root, ignore_errors=True)

    def call(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        connection.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                           headers={"content-type": "application/json", **(headers or {})})
        response = connection.getresponse()
        text = response.read()
        connection.close()
        return response.status, text

    def paint(self, app, **body):
        return self.call("POST", "/v1/images/generations", {"model": "klein-4b", "prompt": "a red fox", **body},
                         {"Authorization": f"Bearer {app}"})

    def metrics(self) -> str:
        status, text = self.call("GET", "/metrics")
        self.assertEqual(status, 200)
        text = text.decode()
        self.assert_valid(text)
        return text

    def assert_valid(self, text: str) -> None:
        typed = set()
        for line in text.rstrip("\n").split("\n"):
            match = re.fullmatch(r"# TYPE (\S+) (counter|gauge|histogram)", line)
            if match:
                typed.add(match[1])
                continue
            if line.startswith("# HELP "):
                continue
            sample = SAMPLE.match(line)
            self.assertIsNotNone(sample, f"ungültige Zeile: {line}")
            family = re.sub(r"_(bucket|sum|count)$", "", sample[1])
            self.assertTrue(sample[1] in typed or family in typed, f"{sample[1]} ohne # TYPE davor")

    def test_every_family_is_there(self):
        text = self.metrics()
        for name in ("imgomat_images_total", "imgomat_paint_seconds", "imgomat_wait_seconds", "imgomat_queue_waiting",
                     "imgomat_painting", "imgomat_worker_errors_total", "imgomat_worker_reachable",
                     "imgomat_downloads_total", "imgomat_files_bytes", "imgomat_http_requests_total",
                     "imgomat_loki_dropped_total", "process_resident_memory_bytes", "process_cpu_seconds_total",
                     "process_start_time_seconds"):
            self.assertIn(f"# TYPE {name} ", text)
        self.assertEqual(value(text, "imgomat_painting", 'model="klein-4b"'), 0)
        self.assertEqual(value(text, "imgomat_queue_waiting", 'client="tonwerk-ai"'), 0)
        self.assertGreater(value(text, "process_resident_memory_bytes"), 0)

    def test_a_picture_counts_with_its_app_and_model(self):
        before = self.metrics()
        done = value(before, "imgomat_images_total", 'client="tonwerk-ai",model="klein-4b",result="done"') or 0
        status, _ = self.paint("tonwerk-ai", n=2)
        self.assertEqual(status, 200)
        text = self.metrics()
        self.assertEqual(value(text, "imgomat_images_total", 'client="tonwerk-ai",model="klein-4b",result="done"'), done + 2)
        self.assertGreaterEqual(value(text, "imgomat_paint_seconds_count", 'model="klein-4b"'), 2)
        self.assertGreaterEqual(value(text, "imgomat_wait_seconds_count", 'client="tonwerk-ai"'), 1)
        self.assertEqual(value(text, 'imgomat_paint_seconds_bucket', 'model="klein-4b",le="+Inf"'),
                         value(text, "imgomat_paint_seconds_count", 'model="klein-4b"'))

    def test_an_unknown_app_becomes_other(self):
        status, _ = self.paint("erfundene-app")
        self.assertEqual(status, 200)
        text = self.metrics()
        self.assertGreaterEqual(value(text, "imgomat_images_total", 'client="other",model="klein-4b",result="done"'), 1)
        self.assertNotIn("erfundene-app", text)

    def test_a_filtered_picture_is_no_worker_error(self):
        before = self.service.metrics.worker_errors.value(kind="runner")
        status, _ = self.paint("3d-o-mat", model="klein-9b", prompt="filter me")
        self.assertEqual(status, 422)
        text = self.metrics()
        self.assertEqual(value(text, "imgomat_images_total", 'client="3d-o-mat",model="klein-9b",result="filtered"'), 1)
        self.assertEqual(self.service.metrics.worker_errors.value(kind="runner"), before)

    def test_a_broken_picture_counts_as_failed_and_runner_error(self):
        before = self.service.metrics.worker_errors.value(kind="runner")
        status, _ = self.paint("img-o-mat", prompt="please fail")
        self.assertEqual(status, 500)
        text = self.metrics()
        self.assertEqual(value(text, "imgomat_images_total", 'client="img-o-mat",model="klein-4b",result="failed"'), 1)
        self.assertEqual(self.service.metrics.worker_errors.value(kind="runner"), before + 1)

    def test_requests_count_by_route_pattern_not_path(self):
        digest = "a" * 64
        self.call("HEAD", f"/v1/files/{digest}")
        self.call("GET", "/gibt-es-nicht")
        text = self.metrics()
        self.assertEqual(value(text, "imgomat_http_requests_total", 'method="HEAD",route="/v1/files/<sha256>",status="404"'), 1)
        self.assertEqual(value(text, "imgomat_http_requests_total", 'method="GET",route="other",status="404"'), 1)
        self.assertNotIn(digest, text)

    def test_log_lines_go_to_loki_with_four_labels(self):
        self.paint("tonwerk-ai", prompt="please fail")
        deadline = time.time() + 5
        while time.time() < deadline and not any(s["stream"]["level"] == "warning"
                                                 for _, push in FakeLoki.pushed for s in push["streams"]):
            time.sleep(0.1)
        streams = [s for _, push in FakeLoki.pushed for s in push["streams"]]
        self.assertTrue(streams, "nichts an Loki geschickt")
        self.assertTrue(all(path == "/loki/api/v1/push" for path, _ in FakeLoki.pushed))
        for stream in streams:
            self.assertEqual(sorted(stream["stream"]), ["app", "host", "level", "source"])
            self.assertEqual(stream["stream"]["app"], "img-o-mat")
            self.assertEqual(stream["stream"]["host"], "probe")
        lines = lambda source: [v[1] for s in streams if s["stream"]["source"] == source for v in s["values"]]
        self.assertTrue(any("male mit klein-4b" in line for line in lines("painter")))
        self.assertTrue(any(s["stream"]["level"] == "warning" and s["stream"]["source"] == "painter" for s in streams))
        # The prompt never goes along.
        self.assertFalse(any("please fail" in v[1] for s in streams for v in s["values"]))
        self.assertGreater(self.service.metrics.loki_sent.value(), 0)


class ShipperTest(unittest.TestCase):
    def test_lines_are_dropped_and_counted_when_loki_is_away(self):
        sent, dropped = Counter("s", "s"), Counter("d", "d")
        shipper = LokiShipper({"url": "http://127.0.0.1:9/", "intervalSeconds": 3600}, sent, dropped, timeout=1)
        shipper.add("eins", "info", "painter")
        shipper.add("zwei", "info", "painter")
        shipper.flush()
        self.assertEqual(dropped.value(), 2)
        self.assertEqual(sent.value(), 0)

    def test_without_url_nothing_is_kept(self):
        sent, dropped = Counter("s", "s"), Counter("d", "d")
        shipper = LokiShipper({}, sent, dropped)
        self.assertFalse(shipper.enabled)
        shipper.add("eins", "info", "painter")
        shipper.flush()
        self.assertEqual((sent.value(), dropped.value()), (0, 0))


class SyntaxTest(unittest.TestCase):
    def test_everything_parses_as_python_3_9(self):
        """probe.py and launch.py run on macOS's /usr/bin/python3 (3.9), the service on Debian's; newer syntax would
        only break there."""
        for path in sorted(glob.glob(os.path.join(ROOT, "service", "*.py")) + glob.glob(os.path.join(ROOT, "worker", "*.py"))):
            with open(path) as file:
                ast.parse(file.read(), path, feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
