"""The service end to end against fake runners (tests/fake): the HTTP API, the queue, the catalogue's rules, files,
LoRAs, downloads and cancelling. The worker runs locally here; over SSH it is the same launch.py.

    python3 -m unittest discover tests
"""

import base64
import contextlib
import io
import hashlib
import http.client
import json
import os
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))

import config as settings  # noqa: E402
import notify  # noqa: E402
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
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "ltxPython": sys.executable, "scripts": FAKE}, out)
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

    def test_the_page_enlarges_downloads_and_deletes_pictures(self):
        status, page, _ = self.call("GET", "/")
        self.assertEqual(status, 200)
        page = page.decode()
        for part in ('id="preview"', 'id="previewDownload" download', 'id="previewDelete"', "data-enlarge"):
            self.assertIn(part, page)

    def test_keeps_a_picture_in_the_gallery(self):
        status, body, _ = self.paint(seed=21)
        self.assertEqual(status, 200, body)
        picture = body["data"][0]["b64_json"]
        # Painting alone keeps nothing, whoever asks (Tonwerk-AI here).
        _, listed, _ = self.call("GET", "/v1/gallery")
        before = {p["name"] for p in listed["pictures"]}
        self.assertNotIn(hashlib.sha256(base64.b64decode(picture)).hexdigest()[:12], " ".join(before))

        keep = {"b64_json": picture, "model": "klein-4b", "seed": 21, "size": "1024x1024", "prompt": "a red fox",
                "creator": "img-o-mat", "contentFilter": False}
        status, saved, _ = self.call("POST", "/v1/gallery", keep, {"X-Llm-App": "img-o-mat"})
        self.assertEqual(status, 201, saved)
        self.assertTrue(saved["created"])
        status, again, _ = self.call("POST", "/v1/gallery", keep, {"X-Llm-App": "img-o-mat"})
        self.assertEqual((status, again["name"], again["created"]), (200, saved["name"], False))

        _, listed, _ = self.call("GET", "/v1/gallery")
        entry = next(p for p in listed["pictures"] if p["name"] == saved["name"])
        self.assertEqual((entry["model"], entry["seed"], entry["prompt"], entry["app"]), ("klein-4b", 21, "a red fox", "img-o-mat"))
        self.assertNotIn("contentFilter", entry)
        self.assertIsNone(listed["disk"])
        self.assertTrue(os.path.exists(os.path.join(self.root, "gallery", saved["name"] + ".jpg")))

        status, data, response = self.call("GET", f"/v1/gallery/{saved['name']}")
        self.assertEqual((status, response.getheader("content-type")), (200, "image/jpeg"))
        self.assertEqual(data, base64.b64decode(picture))
        # On a Mac sips makes the small copy (a new JPEG); elsewhere the picture itself stands in for it.
        status, data, response = self.call("GET", f"/v1/gallery/{saved['name']}?thumb=1")
        self.assertEqual((status, response.getheader("content-type")), (200, "image/jpeg"))
        self.assertTrue(data.startswith(b"\xff\xd8\xff"))
        if not os.path.exists("/usr/bin/sips"):
            self.assertEqual(data, base64.b64decode(picture))
        self.assertTrue(os.path.exists(os.path.join(self.root, "gallery-cache", saved["name"] + ".jpg")))

        status, _, _ = self.call("DELETE", f"/v1/gallery/{saved['name']}")
        self.assertEqual(status, 200)
        self.assertFalse(os.path.exists(os.path.join(self.root, "gallery-cache", saved["name"] + ".jpg")))
        status, body, _ = self.call("GET", f"/v1/gallery/{saved['name']}")
        self.assertEqual(status, 404, body)
        status, body, _ = self.call("DELETE", f"/v1/gallery/{saved['name']}")
        self.assertEqual(status, 404, body)

    def test_the_gallery_keeps_loras_and_the_pictures_it_was_painted_from(self):
        status, body, _ = self.paint(seed=33, prompt="a fox in a jersey")
        picture = body["data"][0]["b64_json"]
        reference = b"\xff\xd8\xff reference for the gallery"
        digest = hashlib.sha256(reference).hexdigest()
        self.assertEqual(self.call("POST", "/v1/files", raw=reference)[0], 201)
        keep = {"b64_json": picture, "model": "klein-4b", "seed": 33, "size": "1024x1024", "prompt": "a fox in a jersey",
                "paintedPrompt": "a red fox wearing a football jersey",
                "loras": [{"name": "trikot", "label": "Trikot", "strength": 0.8}, {"no": "name"}],
                "references": [{"file": digest, "role": "figure", "strength": None},
                               {"file": "0" * 64, "role": "start", "strength": 0.5}]}
        status, saved, _ = self.call("POST", "/v1/gallery", keep, {"X-Llm-App": "img-o-mat"})
        self.assertEqual(status, 201, saved)
        _, listed, _ = self.call("GET", "/v1/gallery")
        entry = next(p for p in listed["pictures"] if p["name"] == saved["name"])
        self.assertEqual(entry["loras"], [{"name": "trikot", "label": "Trikot", "strength": 0.8}])
        self.assertEqual(entry["paintedPrompt"], "a red fox wearing a football jersey")
        # The second was gone from the service: named, without a picture.
        self.assertEqual(entry["references"], [{"role": "figure", "strength": None, "sha256": digest},
                                               {"role": "start", "strength": 0.5}])
        status, data, response = self.call("GET", f"/v1/gallery/ref/{digest}")
        self.assertEqual((status, data, response.getheader("content-type")), (200, reference, "image/jpeg"))
        self.assertEqual(self.call("GET", "/v1/gallery/ref/" + "1" * 64)[0], 404)
        self.assertEqual(self.call("GET", "/v1/gallery/ref/nothex")[0], 404)

        # A second picture from the same reference shares it; it goes with the last of them.
        other = dict(keep, b64_json=base64.b64encode(base64.b64decode(picture) + b"\0").decode())
        status, second, _ = self.call("POST", "/v1/gallery", other)
        self.assertEqual(status, 201, second)
        kept = os.path.join(self.root, "gallery", ".refs", digest + ".jpg")
        self.call("DELETE", f"/v1/gallery/{saved['name']}")
        self.assertTrue(os.path.exists(kept))
        self.call("DELETE", f"/v1/gallery/{second['name']}")
        self.assertFalse(os.path.exists(kept))

    def test_the_gallery_refuses_what_is_no_picture(self):
        cases = [
            ({"b64_json": base64.b64encode(b"GIF89a").decode()}, 400, "kein JPEG"),
            ({"b64_json": "!!!"}, 400, "Base64"),
            ({}, 400, "b64_json"),
        ]
        for body, code, text in cases:
            status, answer, _ = self.call("POST", "/v1/gallery", body)
            self.assertEqual(status, code, answer)
            self.assertIn(text, answer["error"]["message"])
        for name in ("../worker.json", "20261105-093000-zzzzzzzzzzzz"):
            status, _, _ = self.call("GET", "/v1/gallery/" + name)
            self.assertEqual(status, 404)

    def test_the_page_saves_and_shows_the_gallery(self):
        _, page, _ = self.call("GET", "/")
        page = page.decode()
        for part in ('id="previewSave"', 'data-view="galerie"', 'id="galleryGrid"', "'/v1/gallery'"):
            self.assertIn(part, page)

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

    def test_the_page_uploads_pictures_by_content(self):
        content = b"\xff\xd8 picture from the page"
        digest = hashlib.sha256(content).hexdigest()
        status, body, _ = self.call("POST", "/v1/files", raw=content, headers={"content-type": "image/jpeg"})
        self.assertEqual((status, body), (201, {"sha256": digest}))
        status, body, _ = self.call("POST", "/v1/files", raw=content, headers={"content-type": "image/jpeg"})
        self.assertEqual((status, body), (200, {"sha256": digest}))
        self.assertEqual(self.call("HEAD", f"/v1/files/{digest}")[0], 200)
        self.assertEqual(self.call("POST", "/v1/files", raw=b"")[0], 400)
        status, body, _ = self.paint(references=[{"file": digest}])
        self.assertEqual(status, 200, body)
        self.assertEqual(self.last_request()["references"], [{"file": digest, "role": "style", "strength": None}])

    def test_the_page_offers_pictures_to_look_at_and_paint_over(self):
        _, page, _ = self.call("GET", "/")
        page = page.decode()
        for part in ('id="refAdd"', 'id="refFile"', 'id="refStrength"', 'id="previewRef"', 'id="previewReuse"', "'/v1/files'",
                     "role: 'start'", "/v1/gallery/ref/", 'id="previewRefine"'):
            self.assertIn(part, page)

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

    def test_loras_reach_the_runner_with_their_extension(self):
        # mlx infers the format from the extension; files are named by their hash alone.
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import common
        import paint

        folder = os.path.join(self.root, "files")
        os.makedirs(folder, exist_ok=True)
        digest = "ab" * 32
        with open(os.path.join(folder, digest), "wb") as out:
            out.write(safetensors({}))
        saved, common.ROOT = common.ROOT, self.root
        try:
            for _ in range(2):  # the second picture finds the link from the first
                request = {"loras": [{"file": digest, "strength": 1.0}], "references": [{"file": digest, "role": "figure"}]}
                paint.resolve(request)
                self.assertEqual(request["loras"][0]["path"], os.path.join(folder, digest + ".safetensors"))
                self.assertEqual(request["references"][0]["path"], os.path.join(folder, digest))
                with open(request["loras"][0]["path"], "rb") as file:
                    self.assertEqual(file.read(), safetensors({}))
        finally:
            common.ROOT = saved

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

    def test_video_models_wait_apart_from_the_image_models(self):
        # Tonwerk and the other clients list /v1/models and paint from it: no video model may show up there.
        _, listed, _ = self.call("GET", "/v1/models")
        self.assertFalse({"ltx-2.3", "wan-2.2-ti2v-5b"} & {m["id"] for m in listed["data"]})
        status, videos, _ = self.call("GET", "/v1/models?kind=video&fresh=1")
        self.assertEqual(status, 200)
        ltx = next(m for m in videos["data"] if m["id"] == "ltx-2.3")
        self.assertEqual((ltx["kind"], ltx["paints"], ltx["downloaded"]), ("video", True, False))
        wan = next(m for m in videos["data"] if m["id"] == "wan-2.2-ti2v-5b")
        self.assertEqual((wan["paints"], wan["ready"]), (False, False))
        # This Mac has 24 GB; the runner not being installed is no reason, nothing paints videos yet.
        self.assertIn("32 GB", ltx["unavailable"])
        status, body, _ = self.paint(model="ltx-2.3")
        self.assertEqual(status, 400, body)

    def test_downloads_a_video_model_at_its_pinned_commit(self):
        status, state, _ = self.call("POST", "/v1/models/wan-2.2-ti2v-5b/download")
        self.assertEqual(status, 202, state)
        for _ in range(100):
            _, videos, _ = self.call("GET", "/v1/models?kind=video&fresh=1")
            wan = next(m for m in videos["data"] if m["id"] == "wan-2.2-ti2v-5b")
            if wan["download"]["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(wan["download"]["state"], "done", wan["download"])
        self.assertTrue(wan["downloaded"])
        with open(os.path.join(self.root, "last-fetch.json")) as file:
            asked = json.load(file)
        self.assertEqual(asked["repos"], ["Anes1032/Wan2.2-TI2V-5B-mlx-q8"])
        self.assertEqual(asked["revisions"], {"Anes1032/Wan2.2-TI2V-5B-mlx-q8": "9624723c94ddf509832555c45e223a035baa7d1c"})

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

    def test_the_page_lines_up_paintings(self):
        _, page, _ = self.call("GET", "/")
        page = page.decode()
        for part in ("SENT_AHEAD = 2", "Einreihen", "data-drop-job", "auf dieser Seite", "beforeunload"):
            self.assertIn(part, page)

    def test_a_painting_sent_ahead_waits_its_turn(self):
        """The page sends its next painting while one paints; the service keeps it waiting and says so in the stream."""
        painter = self.service.painter
        blocker = self.service.job_from({"model": "klein-4b", "prompt": "slow block"}, "img-o-mat", 50)
        painter.submit(blocker)
        for _ in range(100):
            if painter.running is blocker:
                break
            time.sleep(0.05)
        result = {}
        asking = threading.Thread(target=lambda: result.update(answer=self.call(
            "POST", "/v1/images/generations", {"model": "klein-4b", "prompt": "next fox", "stream": True}, {"x-llm-app": "img-o-mat"})))
        asking.start()
        for _ in range(100):
            if painter.status()["queue"]:
                break
            time.sleep(0.05)
        self.assertEqual([j["app"] for j in painter.status()["queue"]], ["img-o-mat"])
        painter.cancel(blocker)
        asking.join(30)
        # The blocker's pid would mislead the cancel test.
        pid_file = os.path.join(self.root, "slow.pid")
        if os.path.exists(pid_file):
            os.remove(pid_file)
        text = result["answer"][1].decode()
        self.assertLess(text.index("event: queued"), text.index("event: image"))
        self.assertIn('"place": 1', text)
        self.assertTrue(text.rstrip().split("\n\n")[-1].startswith("event: done"))

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
        status, busy, _ = self.call("GET", "/v1/busy")
        self.assertEqual((status, busy["busy"], busy["app"], busy["model"], busy["waiting"]), (200, True, "a", "klein-4b", 2))
        self.assertGreater(busy["memoryGb"], 0)
        for job in (late, urgent, blocker):
            painter.cancel(job)

    def test_tells_onkel_bummel_when_a_picture_is_done(self):
        with FakeOnkelBummel() as bummel:
            token = os.path.join(self.root, "onkel-bummel.token")
            with open(token, "w") as out:
                out.write("ob_test\n")
            self.service.notifier.settings = {"url": bummel.url, "tokenFile": token, "topic": "bilder", "click": "http://img-o-mat:1236/"}
            try:
                status, body, response = self.paint(prompt="a red fox in the snow", n=2)
                self.assertEqual(status, 200, body)
                headers, message = bummel.received.get(timeout=10)
                self.assertEqual(headers["Authorization"], "Bearer ob_test")
                self.assertEqual(headers["Idempotency-Key"], response.getheader("X-Img-Job"))
                self.assertEqual(message["topic"], "bilder")
                self.assertEqual(message["title"], "2 Bilder fertig (FLUX.2 Klein 4B)")
                self.assertIn("tonwerk-ai", message["message"])
                self.assertIn("a red fox in the snow", message["message"])
                self.assertEqual(message["click"], "http://img-o-mat:1236/")
                self.assertEqual(len(message["images"]), 2)
                self.assertTrue(base64.b64decode(message["images"][0]["data"]).startswith(b"\xff\xd8"))

                status, body, _ = self.paint(prompt="please fail")
                self.assertEqual(status, 500)
                _, message = bummel.received.get(timeout=10)
                self.assertEqual((message["title"], message["priority"]), ("Bild fehlgeschlagen (FLUX.2 Klein 4B)", 4))
                self.assertTrue(message["message"].startswith("the model broke"))
                self.assertNotIn("images", message)

                # Asked not to, or an app the config leaves out: nothing.
                self.assertEqual(self.paint(notify=False)[0], 200)
                self.service.notifier.settings["apps"] = ["img-o-mat"]
                self.assertEqual(self.paint()[0], 200)
                time.sleep(0.5)
                self.assertTrue(bummel.received.empty())
            finally:
                self.service.notifier.settings = {}

    def test_onkel_bummel_retries_its_own_trouble_but_not_a_wrong_token(self):
        sleep = notify.time.sleep
        notify.time.sleep = lambda seconds: None
        try:
            with FakeOnkelBummel(answers=[503, 201]) as bummel:
                notifier = server.Notifier({"url": bummel.url, "token": "ob_x"}, lambda line: None)
                self.assertTrue(notifier.send({"topic": "t", "message": "m"}, "k1"))
                self.assertEqual(bummel.received.qsize(), 2)
            with FakeOnkelBummel(answers=[401]) as bummel:
                logged = []
                notifier = server.Notifier({"url": bummel.url, "token": "ob_wrong"}, logged.append)
                self.assertFalse(notifier.send({"topic": "t", "message": "m"}, "k2"))
                self.assertEqual(bummel.received.qsize(), 1)
                self.assertIn("401", logged[0])
            logged = []
            self.assertFalse(server.Notifier({"url": "http://127.0.0.1:9", "token": "ob_x"}, logged.append).send({}, "k3"))
            self.assertIn("nicht erreichbar", logged[0])
            logged = []
            self.assertFalse(server.Notifier({"url": "http://127.0.0.1:9"}, logged.append).send({}, "k4"))
            self.assertIn("kein Token", logged[0])
        finally:
            notify.time.sleep = sleep


class FakeOnkelBummel:
    """POST /api/v1/messages as Onkel Bummel takes it; answers are the status codes to give, then 201."""

    def __init__(self, answers=()):
        self.received = queue.Queue()
        self.answers = list(answers)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.received.put((dict(self.headers), body))
                code = fake.answers.pop(0) if fake.answers else 201
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


if __name__ == "__main__":
    unittest.main()


class SshTest(unittest.TestCase):
    """The same picture over the SSH path: the remote command is only words for launch.py --ssh, and LoRAs travel to
    the worker's own files folder."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="img-o-mat-front-")
        self.remote = tempfile.mkdtemp(prefix="img-o-mat-worker-")
        with open(os.path.join(self.remote, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "ltxPython": sys.executable, "scripts": FAKE}, out)
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

    def test_makes_a_clip_over_ssh(self):
        for repo in ("dgrauet/ltx-2.3-mlx-q8", "mlx-community/gemma-3-12b-it-4bit"):
            os.makedirs(os.path.join(self.remote, "models", "hub", "models--" + repo.replace("/", "--"), "snapshots"))
        service = server.Service(settings.Config({"root": self.root, "worker": {"ssh": "marcel@oi-o-mat"}, "systemMemoryGb": 32}))
        picture = os.path.join(self.root, "card.jpg")
        with open(picture, "wb") as out:
            out.write(b"\xff\xd8\xff\xe0 card")
        digest = hashlib.sha256(b"\xff\xd8\xff\xe0 card").hexdigest()
        service.files.use = lambda d: picture
        job = service.video_job_from({"model": "ltx-2.3", "prompt": "she turns", "images": [{"file": digest, "at": "last"}]}, "test", 0)
        service.painter.submit(job)
        list(job.events(timeout=1))
        self.assertIsNone(job.error, job.error)
        self.assertEqual(job.results[0]["mp4"][4:8], b"ftyp")
        self.assertTrue(os.path.exists(os.path.join(self.remote, "files", digest)))

    def test_keeps_pictures_on_the_worker(self):
        job = self.service.job_from({"model": "klein-4b", "prompt": "a fox"}, "test", 0)
        self.service.painter.submit(job)
        list(job.events(timeout=1))
        picture = base64.b64encode(job.results[0]["jpeg"]).decode()
        saved = self.service.keep({"b64_json": picture, "model": "klein-4b", "seed": 1}, "img-o-mat")
        self.assertTrue(os.path.exists(os.path.join(self.remote, "gallery", saved["name"] + ".jpg")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "gallery")))
        self.assertEqual([p["name"] for p in self.service.gallery.list()["pictures"]], [saved["name"]])
        self.assertEqual(self.service.gallery.picture(saved["name"], thumb=False), job.results[0]["jpeg"])
        self.service.gallery.delete(saved["name"])
        self.assertEqual(self.service.gallery.list()["pictures"], [])

    def test_an_unplugged_disk_keeps_nothing(self):
        with open(os.path.join(self.remote, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "scripts": FAKE, "gallery": "/Volumes/img-o-mat-test-weg/galerie"}, out)
        with self.assertRaises(server.GalleryError) as caught:
            self.service.keep({"b64_json": base64.b64encode(b"\xff\xd8\xff\xe0").decode()}, "img-o-mat")
        self.assertEqual(caught.exception.status, 503)
        self.assertIn("img-o-mat-test-weg ist nicht angeschlossen", caught.exception.message)
        self.assertFalse(os.path.exists("/Volumes/img-o-mat-test-weg"))

    def test_the_gallery_lies_on_the_disk_of_the_models(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import common
        self.assertEqual(common.volume_of("/Volumes/T7/Tonwerk/images/models"), "T7")
        self.assertIsNone(common.volume_of("/Users/marcel/Library/Application Support/img-o-mat/models"))
        with open(os.path.join(self.remote, "worker.json"), "w") as out:
            json.dump({"modelCache": "/Volumes/T7/Tonwerk/images/models"}, out)
        root, common.ROOT = common.ROOT, self.remote
        try:
            values = common.settings()
        finally:
            common.ROOT = root
        self.assertEqual((values["gallery"], values["galleryDisk"]), ("/Volumes/T7/img-o-mat/galerie", "T7"))

    def test_the_key_runs_nothing_else(self):
        out = __import__("subprocess").run(
            [os.path.join(FAKE, "bin", "ssh"), "marcel@oi-o-mat", "img-o-mat rm -rf /"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("Aufruf", out.stderr)


class VideoTest(unittest.TestCase):
    """Clips with LTX-2.3 on a 32 GB Mac: only once downloaded, through the same line as the pictures, with a first
    and a last picture; Tonwerk's model list stays without them."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="img-o-mat-video-")
        with open(os.path.join(cls.root, "worker.json"), "w") as out:
            json.dump({"python": sys.executable, "flux3Python": sys.executable, "ltxPython": sys.executable, "scripts": FAKE}, out)
        os.environ["IMG_O_MAT_SOURCE_WORKER"] = os.path.join(ROOT, "worker")
        config = settings.Config({"root": cls.root, "listen": "127.0.0.1", "port": 0, "systemMemoryGb": 32, "minFreeGb": 0})
        cls.service = server.Service(config)
        server.Handler.service = cls.service
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        # ServiceTest swaps the class-wide service; this class's server must answer with its own.
        server.Handler.service = self.service

    call = ServiceTest.call

    def ltx(self):
        _, videos, _ = self.call("GET", "/v1/models?kind=video&fresh=1")
        return next(m for m in videos["data"] if m["id"] == "ltx-2.3")

    def download(self):
        # As the download button leaves them (fake fetch.py); 81 GB do not fit the test machine's free space check.
        for repo in ("dgrauet/ltx-2.3-mlx-q8", "mlx-community/gemma-3-12b-it-4bit"):
            os.makedirs(os.path.join(self.root, "models", "hub", "models--" + repo.replace("/", "--"), "snapshots"), exist_ok=True)
        self.service.link.forget_probe()

    def upload(self, data: bytes) -> str:
        status, body, _ = self.call("POST", "/v1/files", raw=data, headers={"content-type": "image/jpeg"})
        self.assertIn(status, (200, 201), body)
        return body["sha256"]

    def video(self, **body):
        return self.call("POST", "/v1/videos/generations", {"model": "ltx-2.3", "prompt": "she turns her head", **body},
                         {"X-Llm-App": "img-o-mat"})

    def test_a_00_clip_waits_for_the_download(self):
        model = self.ltx()
        self.assertTrue((model["available"], model["ready"], model["paints"]) == (True, True, True), model)
        status, body, _ = self.video()
        self.assertEqual(status, 409, body)
        self.assertIn("noch nicht heruntergeladen", body["error"]["message"])

    def test_makes_a_clip_from_a_first_and_a_last_picture(self):
        self.download()
        first, last = self.upload(b"\xff\xd8\xff\xe0 first"), self.upload(b"\xff\xd8\xff\xe0 last")
        status, text, response = self.video(stream=True, seed=7, size="768x768", frames=97, quality="fast",
                                            images=[{"file": first, "at": "first"}, {"file": last, "at": "last"}])
        self.assertEqual(status, 200)
        events = [block.split("\n") for block in text.decode().strip().split("\n\n")]
        names = [lines[0].removeprefix("event: ") for lines in events]
        self.assertEqual(names[-2:], ["video", "done"], names)
        clip = json.loads(events[-2][1].removeprefix("data: "))
        self.assertEqual(base64.b64decode(clip["b64_mp4"])[4:8], b"ftyp")
        self.assertEqual(clip["seed"], 7)
        self.assertEqual(clip["peakMemoryGb"], 12.5)
        with open(os.path.join(self.root, "last-video.json")) as file:
            asked = json.load(file)
        self.assertEqual((asked["model"], asked["textEncoder"]), ("dgrauet/ltx-2.3-mlx-q8", "mlx-community/gemma-3-12b-it-4bit"))
        self.assertEqual((asked["width"], asked["height"], asked["frames"], asked["fps"], asked["quality"]), (768, 768, 97, 24, "fast"))
        self.assertEqual([(i["file"], i["at"]) for i in asked["images"]], [(first, "first"), (last, "last")])
        # Tonwerk's list and the picture route know nothing of it.
        _, listed, _ = self.call("GET", "/v1/models")
        self.assertNotIn("ltx-2.3", {m["id"] for m in listed["data"]})

    def test_a_plain_request_gets_the_clip_and_a_broken_one_says_why(self):
        self.download()
        status, body, _ = self.video()
        self.assertEqual(status, 200, body)
        self.assertTrue(body["data"][0]["b64_mp4"])
        self.assertEqual(body["data"][0]["peakMemoryGb"], 12.5)
        status, body, _ = self.video(prompt="please fail")
        self.assertEqual(status, 500)
        self.assertIn("ohne Video", body["error"]["message"])

    def test_checks_what_ltx_takes(self):
        self.download()
        for asked, part in (({"frames": 100}, "8·k+1"), ({"size": "1000x768"}, "durch 64"), ({"quality": "best"}, "quality"),
                            ({"images": [{"file": "a" * 64}, {"file": "b" * 64}]}, "ein Bild am Anfang"),
                            ({"model": "wan-2.2-ti2v-5b"}, "Unbekanntes Videomodell")):
            status, body, _ = self.video(**asked)
            self.assertEqual(status, 400, (asked, body))
            self.assertIn(part, body["error"]["message"])

    def prepare_minimax(self):
        self.service.config["systemMemoryGb"] = 64
        self.addCleanup(self.service.config.__setitem__, "systemMemoryGb", 32)
        config_file = os.path.join(self.root, "worker.json")
        with open(config_file) as file:
            original = json.load(file)
        with open(config_file, "w") as file:
            json.dump({**original, "minimaxPython": sys.executable}, file)
        def restore_worker():
            with open(config_file, "w") as file:
                json.dump(original, file)
            self.service.link.forget_probe()
        self.addCleanup(restore_worker)
        scripts = os.path.join(self.root, "minimax-h3-mlx", "scripts")
        os.makedirs(scripts, exist_ok=True)
        with open(os.path.join(scripts, "generate.py"), "w") as file:
            file.write("# fake installation")
        streaming = os.path.join(self.root, "minimax-h3-streaming", "minimax_h3_mlx")
        os.makedirs(streaming, exist_ok=True)
        with open(os.path.join(streaming, "streaming.py"), "w") as file:
            file.write("# fake streaming installation")
        model = self.service.catalogue.find_video("minimax-h3-mlx-4bit")
        for repo in (model["repo"], *model["moreRepos"]):
            os.makedirs(os.path.join(self.root, "models", "hub", "models--" + repo.replace("/", "--"), "snapshots"), exist_ok=True)
        self.service.link.forget_probe()
        return model

    def test_minimax_clip_uses_its_runner_and_keyframes(self):
        self.prepare_minimax()
        first, last = self.upload(b"\xff\xd8\xff\xe0 h3 first"), self.upload(b"\xff\xd8\xff\xe0 h3 last")
        status, body, _ = self.video(model="minimax-h3-mlx-4bit", duration=10, size="1344x768", audio=False,
                                     images=[{"file": first}, {"file": last, "at": "last"}])
        self.assertEqual(status, 200, body)
        with open(os.path.join(self.root, "last-video.json")) as file:
            asked = json.load(file)
        self.assertEqual((asked["runner"], asked["checkpoint"], asked["duration"], asked["frames"], asked["audio"]),
                         ("minimax-h3-mlx", "MiniMaxAI/MiniMax-H3", 10, 243, False))
        self.assertEqual([i["at"] for i in asked["images"]], ["first", "last"])
        _, models, _ = self.call("GET", "/v1/models")
        self.assertNotIn("minimax-h3-mlx-4bit", [m["id"] for m in models["data"]])

    def test_minimax_validates_its_own_limits_and_full_pipeline_memory(self):
        self.prepare_minimax()
        for asked in ({"duration": 4}, {"duration": 16}, {"duration": None}, {"duration": "invalid"},
                      {"duration": float("nan")}, {"frames": 121},
                      {"size": "512x512"}, {"size": "1344x1344"}):
            status, body, _ = self.video(model="minimax-h3-mlx-4bit", **asked)
            self.assertEqual(status, 400, (asked, body))
        self.service.config["systemMemoryGb"] = 24
        status, body, _ = self.video(model="minimax-h3-mlx-4bit")
        self.assertEqual(status, 501, body)
        self.assertIn("32 GB", body["error"]["message"])

    def test_minimax_32gb_profile_limits_requests_and_selects_phased_loading(self):
        self.prepare_minimax()
        self.service.config["systemMemoryGb"] = 32
        _, models, _ = self.call("GET", "/v1/models?kind=video&fresh=1")
        model = next(m for m in models["data"] if m["id"] == "minimax-h3-mlx-4bit")
        self.assertTrue(model["ready"] and model["lowRam"], model)
        for parameters in ({"duration": 10}, {"size": "1344x768"}):
            status, body, _ = self.video(model=model["id"], **parameters)
            self.assertEqual(status, 400, body)
            self.assertIn("32-GB-Modus", body["error"]["message"])
        status, body, _ = self.video(model=model["id"], lowRam=False)
        self.assertEqual(status, 200, body)
        with open(os.path.join(self.root, "last-video.json")) as file:
            asked = json.load(file)
        self.assertTrue(asked["lowRam"])
        self.assertEqual((asked["width"], asked["height"], asked["duration"]), (768, 768, 5))

    def test_minimax_download_includes_only_required_upstream_components(self):
        model = self.prepare_minimax()
        self.service.downloads.start(model)
        until = time.time() + 5
        while time.time() < until and self.service.downloads.state(model["id"])["state"] not in ("done", "failed"):
            time.sleep(0.02)
        self.assertEqual(self.service.downloads.state(model["id"])["state"], "done")
        with open(os.path.join(self.root, "last-fetch.json")) as file:
            asked = json.load(file)
        self.assertEqual(asked["repos"], [model["repo"], *model["moreRepos"]])
        self.assertIn("FL2VA/text_encoder/config.json", asked["allowPatterns"][model["checkpoint"]])
        self.assertNotIn("FL2VA/text_encoder/*", asked["allowPatterns"][model["checkpoint"]])
        self.assertEqual(asked["allowPatterns"][model["textEncoder"]][0], "text_encoder.safetensors")
        self.assertNotIn("FL2VA/transformer/*", asked["allowPatterns"][model["checkpoint"]])

    def test_minimax_builds_the_offline_command_and_parses_progress(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import video
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "a" * 64), "wb") as file:
                file.write(b"image")
            request = {"model": "quant", "checkpoint": "base", "textEncoder": "encoder", "prompt": "fox", "duration": 15,
                       "quality": "high", "seed": 7, "width": 1344, "height": 768, "lowRam": True,
                       "images": [{"file": "a" * 64, "at": "last"}]}
            with patch.object(video, "snapshot", side_effect=lambda repo: "/cache/" + repo):
                args = video.minimax_command(request, {"files": folder, "minimaxSource": "/runner", "minimaxStreamingSource": "/stream"}, "/out.mp4")
            for flag, value in (("--checkpoint", "/cache/base/FL2VA"), ("--transformer", "/cache/quant"),
                                ("--steps", "17"), ("--duration", "15"), ("--anchor", "last"),
                                ("--quantized-text-encoder", "/cache/encoder"), ("--source", "/runner"), ("--streaming-source", "/stream")):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertEqual(args[:3], [sys.executable, "-u", os.path.join(ROOT, "worker", "minimax_generate.py")])
            self.assertIn("--low-ram", args)
        self.assertEqual(video.progress("  step 4/8  123s", True, 0), (0.5, False))
        self.assertEqual(video.progress("  step 8/8  123s", True, 0), (0.95, True))
        self.assertIsNone(video.progress("loading model", True, 0))

    def test_minimax_launcher_uses_the_separate_python(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import launch
        values = {"minimaxPython": "/h3/python", "ltxPython": "/ltx/python", "scripts": "/scripts"}
        with patch.object(launch.sys, "argv", ["launch.py", "video", "minimax-h3-mlx"]), \
             patch.object(launch.common, "settings", return_value=values), \
             patch.object(launch, "run", return_value=0) as run:
            self.assertEqual(launch.main(), 0)
            run.assert_called_once_with("video.py", "/h3/python", "/scripts")

    def test_fetch_applies_the_component_filter_to_huggingface(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import fetch
        from types import SimpleNamespace
        from unittest.mock import Mock
        download = Mock()
        asked = {"repos": ["quant", "base"], "allowPatterns": {"base": ["FL2VA/text_encoder/*", "LICENSE"]}}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(fetch.common, "settings", return_value={"modelCache": folder}), \
             patch.object(fetch.common, "prepare_environment"), \
             patch.object(fetch.sys, "stdin", io.StringIO(json.dumps(asked))), \
             patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.main(), 0)
        self.assertEqual(download.call_args_list[0].kwargs, {"revision": None, "allow_patterns": None})
        self.assertEqual(download.call_args_list[1].kwargs,
                         {"revision": None, "allow_patterns": ["FL2VA/text_encoder/*", "LICENSE"]})

    def test_minimax_worker_runs_the_cli_and_returns_the_clip(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import video
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "scripts"))
            with open(os.path.join(folder, "scripts", "generate.py"), "w") as file:
                file.write("import sys\nfrom pathlib import Path\n"
                           "assert sys.argv[-2:] == ['--', '-a fox']\n"
                           "Path(sys.argv[sys.argv.index('--output') + 1]).write_bytes(b'fake mp4')\n"
                           "print('  step 8/8  100s', flush=True)\n"
                           "print('MINIMAX_PEAK_GB=8.25', flush=True)\n")
            package = os.path.join(folder, "minimax_h3_mlx")
            os.makedirs(package)
            with open(os.path.join(package, "__init__.py"), "w") as file:
                file.write("")
            with open(os.path.join(package, "text_encoder.py"), "w") as file:
                file.write("class MiniMaxH3TextEncoder: pass\n")
            with open(os.path.join(folder, "config.json"), "w") as file:
                json.dump({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
                           "text_encoder": {"hidden": 5120, "layers": 50, "heads": 64, "kv_heads": 8,
                                            "head_dim": 128, "intermediate": 25600, "theta": 5000000.0}}, file)
            request = {"runner": "minimax-h3-mlx", "model": "quant", "checkpoint": "base", "textEncoder": "encoder", "duration": 5,
                       "prompt": "-a fox", "seed": 0, "width": 768, "height": 768}
            output = io.StringIO()
            with patch.object(video.common, "settings", return_value={"minimaxSource": folder}), \
                 patch.object(video.common, "prepare_environment"), \
                 patch.object(video, "snapshot", return_value=folder), \
                 patch.object(video, "with_ai_note", side_effect=lambda source, *a, **kw: source), \
                 patch.object(video.sys, "stdin", io.StringIO(json.dumps(request))), \
                 patch.dict(os.environ), contextlib.redirect_stdout(output):
                self.assertEqual(video.main(), 0)
            events = [json.loads(line[len("IMGOMAT "):]) for line in output.getvalue().splitlines() if line.startswith("IMGOMAT ")]
            self.assertTrue(any(e.get("stage") == "painting" and e["fraction"] == 0.95 for e in events))
            self.assertEqual(base64.b64decode(events[-1]["mp4"]), b"fake mp4")
            self.assertEqual(events[-1]["peakMemoryGb"], 8.25)

    def test_muting_a_clip_copies_only_the_video_stream(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import video
        with patch.object(video.shutil, "which", return_value="/ffmpeg"), patch.object(video.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(video.with_ai_note("/in", "/out", "H3", "test", audio=False), "/out")
            args = run.call_args[0][0]
            self.assertEqual(args[args.index("-map") + 1], "0:v:0")
            run.return_value.returncode = 1
            run.return_value.stderr = "failed"
            with self.assertRaises(RuntimeError):
                video.with_ai_note("/in", "/out", "H3", "test", audio=False)

    def test_the_worker_builds_the_ltx_command(self):
        sys.path.insert(0, os.path.join(ROOT, "worker"))
        import video
        saved = video.snapshot, video.memory_gb
        video.snapshot = lambda repo: f"/cache/{repo}"
        video.memory_gb = lambda: 32.0
        work = tempfile.mkdtemp()
        files = os.path.join(work, "files")
        os.makedirs(files)
        for name in ("a" * 64, "b" * 64):
            open(os.path.join(files, name), "wb").close()
        try:
            arguments = video.command({"model": "m", "textEncoder": "g", "prompt": "p", "seed": 1, "width": 768,
                                       "height": 512, "frames": 49, "fps": 24, "quality": "high", "audio": False,
                                       "images": [{"file": "a" * 64, "at": "first"}, {"file": "b" * 64, "at": "last", "strength": 0.8}]},
                                      {"files": files}, os.path.join(work, "clip.mp4"))
        finally:
            video.snapshot, video.memory_gb = saved
        text = " ".join(arguments)
        for part in ("--model /cache/m", "--gemma /cache/g", "--two-stage", "--frames 49", "--frame-rate 24", "--low-ram", "--no-audio"):
            self.assertIn(part, text)
        images = [arguments[i + 1:i + 5] for i, a in enumerate(arguments) if a == "--image"]
        self.assertEqual([i[1:] for i in images], [["0", "1.0", "--image"], ["last", "0.8", "guide"]])
        self.assertTrue(all(os.path.islink(i[0]) and i[0].endswith(".jpg") for i in images))
        shutil.rmtree(work, ignore_errors=True)
