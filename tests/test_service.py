"""The service end to end against fake runners (tests/fake): the HTTP API, the queue, the catalogue's rules, files,
LoRAs, downloads and cancelling. The worker runs locally here; over SSH it is the same launch.py.

    python3 -m unittest discover tests
"""

import base64
import hashlib
import http.client
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))

import config as settings  # noqa: E402
import server  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "fake")


def safetensors(header: dict, payload: bytes = b"\0" * 16) -> bytes:
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + payload


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="img-o-mat-test-")
        with open(os.path.join(cls.root, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "scripts": FAKE}, out)
        os.environ["IMG_O_MAT_SOURCE_WORKER"] = os.path.join(ROOT, "worker")
        # A Mac with 24 GB: Krea 2 and FLUX 3 are not offered.
        config = settings.Config({"root": cls.root, "listen": "127.0.0.1", "port": 0, "systemMemoryGb": 24, "minFreeGb": 0})
        server.Handler.service = server.Service(config)
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.service = server.Handler.service

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.root, ignore_errors=True)

    def call(self, method, path, body=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        connection.request(method, path, body=data, headers={"content-type": "application/json", **(headers or {})})
        response = connection.getresponse()
        text = response.read()
        connection.close()
        return response.status, (json.loads(text) if text and response.getheader("content-type", "").startswith("application/json") else text), response

    def model(self, model_id):
        status, body, _ = self.call("GET", "/v1/models?fresh=1")
        self.assertEqual(status, 200)
        return next(m for m in body["data"] if m["id"] == model_id)

    def last_request(self):
        with open(os.path.join(self.root, "last-request.json")) as file:
            return json.load(file)

    def paint(self, **body):
        return self.call("POST", "/v1/images/generations", {"model": "klein-4b", "prompt": "a red fox", **body},
                         {"Authorization": "Bearer tonwerk-ai", "X-Img-Priority": "0"})

    def test_models_say_what_this_mac_can(self):
        klein = self.model("klein-4b")
        self.assertTrue(klein["available"])
        self.assertTrue(klein["loras"])
        self.assertEqual(klein["references"], 4)
        krea = self.model("krea-2-turbo")
        self.assertFalse(krea["available"])
        self.assertIn("32 GB", krea["unavailable"])
        self.assertTrue(krea["contentFilter"])
        self.assertTrue(krea["startImage"])

    def test_paints_a_picture(self):
        status, body, response = self.paint(seed=7)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["data"][0]["seed"], 7)
        self.assertTrue(base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\xff\xd8"))
        self.assertTrue(response.getheader("X-Img-Job"))
        request = self.last_request()
        self.assertEqual(request["model"], "flux2-klein-4b")
        self.assertEqual(request["quantize"], 8)
        self.assertEqual(request["creator"], "tonwerk-ai")
        self.assertFalse(request["contentFilter"])

    def test_n_pictures_count_their_seeds_up(self):
        status, body, _ = self.paint(seed=10, n=3)
        self.assertEqual(status, 200, body)
        self.assertEqual([d["seed"] for d in body["data"]], [10, 11, 12])

    def test_the_filter_cannot_be_switched_off(self):
        status, body, _ = self.paint(model="klein-9b", prompt="filter me", contentFilter=False)
        self.assertEqual(status, 422, body)
        self.assertIn("content filter", body["error"]["message"])
        self.assertTrue(self.last_request()["contentFilter"])

    def test_streams_progress_and_the_picture(self):
        status, text, _ = self.paint(stream=True, seed=3)
        self.assertEqual(status, 200)
        events = [block.split("\n") for block in text.decode().strip().split("\n\n")]
        names = [lines[0].removeprefix("event: ") for lines in events]
        self.assertEqual(names[0], "started")
        self.assertIn("progress", names)
        self.assertEqual(names[-2:], ["image", "done"])
        image = json.loads(events[-2][1].removeprefix("data: "))
        self.assertEqual(image["seed"], 3)

    def test_a_broken_picture_says_why(self):
        status, body, _ = self.paint(prompt="please fail")
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["message"], "the model broke")

    def test_refuses_what_a_model_cannot(self):
        cases = [
            ({"model": "nope"}, 400, "Unbekanntes Modell"),
            ({"model": "krea-2-turbo"}, 501, "32 GB"),
            ({"prompt": ""}, 400, "Prompt"),
            ({"size": "1000x1000"}, 400, "16"),
            ({"n": 9}, 400, "n geht"),
            ({"loras": [{"name": "nichtda"}]}, 400, "gibt es nicht"),
            ({"references": [{"file": "0" * 64, "role": "style"}]}, 400, "nicht da"),
        ]
        for extra, expected, words in cases:
            with self.subTest(extra=extra):
                status, body, _ = self.paint(**extra)
                self.assertEqual(status, expected, body)
                self.assertIn(words, body["error"]["message"])

    def test_references_and_start_pictures_follow_the_model(self):
        content = b"\xff\xd8 reference"
        digest = hashlib.sha256(content).hexdigest()
        self.assertEqual(self.call("HEAD", f"/v1/files/{digest}")[0], 404)
        self.assertEqual(self.call("PUT", f"/v1/files/{digest}", raw=content)[0], 201)
        self.assertEqual(self.call("HEAD", f"/v1/files/{digest}")[0], 200)
        self.assertEqual(self.call("PUT", "/v1/files/" + "a" * 64, raw=content)[0], 400)

        status, body, _ = self.paint(references=[{"file": digest, "role": "figure"}])
        self.assertEqual(status, 200, body)
        self.assertEqual(self.last_request()["references"], [{"file": digest, "role": "figure", "strength": None}])
        status, body, _ = self.paint(references=[{"file": digest, "role": "start", "strength": 0.5}])
        self.assertEqual(status, 400)
        self.assertIn("Startbild", body["error"]["message"])
        status, body, _ = self.paint(model="z-image-turbo", references=[{"file": digest, "role": "style"}])
        self.assertEqual(status, 400)
        self.assertIn("keine Referenzbilder", body["error"]["message"])

    def test_loras_from_the_library(self):
        # FLUX.2's fused single-block layer, 3072 wide: Klein 4B.
        content = safetensors({"single_blocks.0.to_qkv_mlp_proj.lora_A.weight": {"dtype": "F16", "shape": [16, 3072], "data_offsets": [0, 16]},
                               "__metadata__": {"ss_trigger": "fox style"}})
        status, info, _ = self.call("PUT", "/v1/loras/Fuchs", raw=content)
        self.assertEqual(status, 200, info)
        self.assertEqual(info["model"], "klein-4b")
        self.assertEqual(info["trigger"], "fox style")
        status, info, _ = self.call("PATCH", "/v1/loras/Fuchs", {"label": "Fuchsstil", "strength": 5})
        self.assertEqual(info["label"], "Fuchsstil")
        self.assertEqual(info["strength"], 2)
        self.assertEqual(self.call("PUT", "/v1/loras/kaputt", raw=b"not safetensors")[0], 400)

        status, body, _ = self.paint(loras=[{"name": "Fuchs"}])
        self.assertEqual(status, 200, body)
        self.assertEqual(self.last_request()["loras"], [{"file": hashlib.sha256(content).hexdigest(), "strength": 2.0}])
        status, body, _ = self.paint(model="z-image-turbo", loras=[{"name": "Fuchs"}])
        self.assertEqual(status, 400)
        self.assertIn("für klein-4b", body["error"]["message"])
        status, body, _ = self.paint(model="flux3-klein", loras=[{"name": "Fuchs"}])
        self.assertIn(status, (400, 501))

        listing = self.call("GET", "/v1/loras")[1]
        self.assertEqual([l["name"] for l in listing["loras"]], ["Fuchs"])
        self.assertTrue(any(s["name"] == "krea-neondrip" for s in listing["suggestions"]))
        self.assertEqual(self.call("DELETE", "/v1/loras/Fuchs")[0], 200)
        self.assertEqual(self.call("DELETE", "/v1/loras/Fuchs")[0], 404)

    def test_downloads_a_model(self):
        self.assertFalse(self.model("z-image-turbo")["downloaded"])
        status, state, _ = self.call("POST", "/v1/models/z-image-turbo/download")
        self.assertEqual(status, 202, state)
        for _ in range(100):
            model = self.model("z-image-turbo")
            if model["download"]["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(model["download"]["state"], "done", model["download"])
        self.assertTrue(model["downloaded"])

    def test_cancel_kills_the_painting_process(self):
        result = {}

        def ask():
            result["answer"] = self.paint(prompt="slow fox", stream=True)

        asking = threading.Thread(target=ask)
        asking.start()
        pid_file = os.path.join(self.root, "slow.pid")
        for _ in range(100):
            if os.path.exists(pid_file):
                break
            time.sleep(0.1)
        with open(pid_file) as file:
            pid = int(file.read())
        os.remove(pid_file)
        job = self.service.painter.running
        self.assertIsNotNone(job)
        self.assertEqual(self.call("DELETE", f"/v1/jobs/{job.id}")[0], 200)
        asking.join(30)
        self.assertIn(b"abgebrochen", result["answer"][1])
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("the painting process is still running")

    def test_priority_goes_first(self):
        painter = self.service.painter
        blocker = self.service.job_from({"model": "klein-4b", "prompt": "slow block"}, "a", 50)
        painter.submit(blocker)
        for _ in range(100):
            if painter.running is blocker:
                break
            time.sleep(0.05)
        late = self.service.job_from({"model": "klein-4b", "prompt": "x"}, "gate-app", 50)
        urgent = self.service.job_from({"model": "klein-4b", "prompt": "y"}, "tonwerk-ai", 0)
        painter.submit(late)
        painter.submit(urgent)
        self.assertEqual([j["app"] for j in painter.status()["queue"]], ["tonwerk-ai", "gate-app"])
        for job in (late, urgent, blocker):
            painter.cancel(job)


if __name__ == "__main__":
    unittest.main()


class SshTest(unittest.TestCase):
    """The same picture over the SSH path: the remote command is only words for launch.py --ssh, and LoRAs travel to
    the worker's own files folder."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="img-o-mat-front-")
        self.remote = tempfile.mkdtemp(prefix="img-o-mat-worker-")
        with open(os.path.join(self.remote, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "scripts": FAKE}, out)
        self.environment = dict(os.environ)
        os.environ["IMG_O_MAT_SOURCE_WORKER"] = os.path.join(ROOT, "worker")
        os.environ["IMG_O_MAT_WORKER_ROOT"] = self.remote
        os.environ["PATH"] = os.path.join(FAKE, "bin") + os.pathsep + os.environ["PATH"]
        self.service = server.Service(settings.Config({"root": self.root, "worker": {"ssh": "marcel@oi-o-mat"}}))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.environment)
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.remote, ignore_errors=True)

    def test_paints_over_ssh_with_a_lora(self):
        content = safetensors({"__metadata__": {"ss_base_model_version": "flux2_klein_4b"}})
        info = self.service.loras.save("Stil", __import__("io").BytesIO(content), len(content), {})
        self.assertEqual(info["model"], "klein-4b")
        self.assertTrue(self.service.models(fresh=True)[0]["available"])
        job = self.service.job_from({"model": "klein-4b", "prompt": "a fox", "loras": [{"name": "Stil"}]}, "test", 0)
        self.service.painter.submit(job)
        list(job.events(timeout=1))
        self.assertIsNone(job.error, job.error)
        self.assertEqual(len(job.results), 1)
        self.assertTrue(os.path.exists(os.path.join(self.remote, "files", info["sha256"])))

    def test_the_key_runs_nothing_else(self):
        out = __import__("subprocess").run(
            [os.path.join(FAKE, "bin", "ssh"), "marcel@oi-o-mat", "img-o-mat rm -rf /"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("Aufruf", out.stderr)
