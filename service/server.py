"""img-o-mat: the image models (FLUX.2 Klein, Z-Image Turbo, Krea 2 Turbo, FLUX 3 Klein) as a service for every app on
Marcel's Macs, with a page to manage models and LoRAs and to paint by hand.

It runs on mac-mini-von-marcel and has the pictures painted on oi-o-mat over SSH (worker_link.py), where the models
lie on the T7. Other apps reach it through the LLM gate (scripts repository, port 1235), which decides who may use
which model and keeps it out of Tonwerk's way; Tonwerk-AI asks directly, first in line.

    python3 service/server.py         (LaunchAgent de.bvelop.img-o-mat, install.sh)

The API, OpenAI's images API where it has one (README, "Schnittstelle"):
    GET    /v1/models                       the models, with what the Mac that paints can do
    POST   /v1/models/<id>/download         start downloading one on the Mac that paints
    POST   /v1/images/generations           paint; "stream": true for progress as server-sent events
    DELETE /v1/jobs/<id>                    cancel a picture
    HEAD   /v1/files/<sha256>, PUT ...      reference pictures (and LoRAs not in the library), once per file
    GET    /v1/loras, PUT/PATCH/DELETE /v1/loras/<name>, POST /v1/loras/fetch
    GET    /status                          what runs and waits, the worker, the downloads
    GET    /                                the page
Plain Python 3.9, nothing to install: macOS's own python3 runs it.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import select
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as settings  # noqa: E402
from catalogue import Catalogue, max_references, takes_loras, takes_start_image  # noqa: E402
from downloads import Downloads  # noqa: E402
from files import FileError, Files  # noqa: E402
from loras import MAX_PER_PICTURE, MAX_STRENGTH, SUGGESTIONS, LoraError, Loras  # noqa: E402
from painter import Job, Painter  # noqa: E402
from worker_link import WorkerLink  # noqa: E402

MAX_PROMPT = 2000
MAX_N = 4
ROLES = ("style", "figure", "place", "start")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
# Tonwerk-AI waits for a cover it asked for; other apps through the gate come after it.
DEFAULT_PRIORITY = 50
PING_SECONDS = 15


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class Service:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.root, exist_ok=True)
        self.catalogue = Catalogue(config)
        self.link = WorkerLink(config, log)
        self.files = Files(config, log)
        self.loras = Loras(config, self.catalogue, log)
        self.downloads = Downloads(self.link, log)
        self.painter = Painter(config, self.link, log)
        self.jobs: dict[str, Job] = {}
        threading.Thread(target=self._housekeeping, daemon=True, name="housekeeping").start()

    def _housekeeping(self) -> None:
        while True:
            self.files.prune()
            for job_id, job in list(self.jobs.items()):
                if job.done and time.time() - job.created > 3600:
                    self.jobs.pop(job_id, None)
            time.sleep(3600)

    def probe(self, fresh: bool = False):
        return self.link.probe(self.catalogue.all_repos(), fresh)

    def models(self, fresh: bool = False) -> list[dict]:
        # The downloads' states before the probe: a download that ends meanwhile then still shows as running rather
        # than done but missing.
        states = {m["id"]: self.downloads.state(m["id"]) for m in self.catalogue.models}
        probe = self.probe(fresh)
        return [self.catalogue.describe(m, probe, self.link.probe_error, states[m["id"]]) for m in self.catalogue.models]

    def status(self) -> dict:
        probe = self.probe()
        return {
            "worker": {"name": self.link.name, "probe": probe, "error": self.link.probe_error},
            "painter": self.painter.status(),
            "downloads": {k: v for k, v in self.downloads.states.items()},
        }

    def job_from(self, body: dict, app: str, priority: int) -> Job:
        """Checks a generation request against the model's rules (the client's dialog checks them too)."""
        model = self.catalogue.find(str(body.get("model") or ""))
        if model is None:
            known = ", ".join(m["id"] for m in self.catalogue.models)
            raise RequestError(400, f"Unbekanntes Modell {body.get('model')!r}; es gibt {known}.")
        if reason := self.catalogue.unavailable(model, self.probe(), self.link.probe_error):
            raise RequestError(501, reason)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise RequestError(400, "Der Prompt fehlt.")
        if len(prompt) > MAX_PROMPT:
            raise RequestError(400, f"Der Prompt ist länger als {MAX_PROMPT} Zeichen.")
        if body.get("response_format") not in (None, "b64_json"):
            raise RequestError(400, "Nur response_format b64_json; Bilder bleiben nicht auf dem Server.")
        n = int(body.get("n") or 1)
        if not 1 <= n <= MAX_N:
            raise RequestError(400, f"n geht von 1 bis {MAX_N}.")
        size = str(body.get("size") or "1024x1024")
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        width, height = (int(match[1]), int(match[2])) if match else (0, 0)
        if not (256 <= width <= 2048 and 256 <= height <= 2048 and width % 16 == 0 and height % 16 == 0):
            raise RequestError(400, "size ist BreitexHöhe, je 256 bis 2048 und durch 16 teilbar.")
        seed = body.get("seed")
        seed = random.randrange(1 << 31) if seed is None else int(seed)
        steps = body.get("steps")
        if steps is not None and not 1 <= int(steps) <= 50:
            raise RequestError(400, "steps geht von 1 bis 50.")

        files: dict[str, str] = {}
        loras = []
        asked = body.get("loras") or []
        if asked and not takes_loras(model):
            raise RequestError(400, f"{model['label']} nimmt keine LoRAs.")
        if len(asked) > MAX_PER_PICTURE:
            raise RequestError(400, f"Höchstens {MAX_PER_PICTURE} LoRAs je Bild.")
        for lora in asked:
            strength = lora.get("strength")
            if lora.get("name"):
                info = self.loras.describe(str(lora["name"]))
                if info is None:
                    raise RequestError(400, f"LoRA {lora['name']} gibt es nicht.")
                if info["model"] and info["model"] != model["id"]:
                    raise RequestError(400, f"LoRA {lora['name']} ist für {info['model']}, nicht für {model['id']}.")
                if not info["model"]:
                    raise RequestError(400, f"Für LoRA {lora['name']} ist kein Modell bekannt; auf der Seite eintragen.")
                files[info["sha256"]] = self.loras.path_for(info["name"])
                loras.append({"file": info["sha256"], "strength": info["strength"] if strength is None else float(strength)})
            else:
                digest = str(lora.get("file") or "")
                files[digest] = self.files_use(digest)
                loras.append({"file": digest, "strength": 1.0 if strength is None else float(strength)})
        for lora in loras:
            lora["strength"] = max(0.0, min(lora["strength"], MAX_STRENGTH))

        references = []
        for reference in body.get("references") or []:
            role = reference.get("role") or "style"
            if role not in ROLES:
                raise RequestError(400, f"Unbekannte Rolle {role}; es gibt {', '.join(ROLES)}.")
            digest = str(reference.get("file") or "")
            files[digest] = self.files_use(digest)
            strength = reference.get("strength")
            references.append({"file": digest, "role": role,
                               "strength": None if strength is None else max(0.1, min(float(strength), 0.9))})
        starts = [r for r in references if r["role"] == "start"]
        looks = [r for r in references if r["role"] != "start"]
        if starts and (not takes_start_image(model) or len(starts) > 1 or looks):
            raise RequestError(400, f"{model['label']} malt über kein Startbild." if not takes_start_image(model)
                               else "Ein Startbild, und dann keine weiteren Referenzen.")
        if len(looks) > max_references(model):
            raise RequestError(400, f"{model['label']} nimmt {max_references(model) or 'keine'} Referenzbilder.")

        request = {
            "prompt": prompt, "n": n, "width": width, "height": height, "seed": seed, "steps": steps,
            "loras": loras, "references": references, "files": files,
            "creator": str(body.get("creator") or "")[:60] or None,
        }
        job = Job(model, request, app, priority)
        self.jobs[job.id] = job
        return job

    def files_use(self, digest: str) -> str:
        try:
            return self.files.use(digest)
        except FileError as error:
            raise RequestError(error.status, error.message)


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def answer_of(job: Job) -> dict:
    return {
        "created": int(job.created),
        "model": job.model["id"],
        "data": [
            {
                "b64_json": base64.b64encode(result["jpeg"]).decode("ascii"),
                "revised_prompt": result["revisedPrompt"],
                "seed": result["seed"],
                "loadSeconds": result["loadSeconds"],
                "paintSeconds": result["paintSeconds"],
                "peakMemoryGb": result["peakMemoryGb"],
            }
            for result in job.results
        ],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "img-o-mat"
    protocol_version = "HTTP/1.1"
    service: Service

    def log_message(self, format, *args):  # the access log would drown the service's own lines
        pass

    # --- plumbing ------------------------------------------------------------------------------------------------

    def send_json(self, status: int, value, headers: dict | None = None) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        """OpenAI's error shape, so clients show the message."""
        self.send_json(status, {"error": {"message": message, "type": "img_o_mat", "code": status}})

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1 << 20:
            raise RequestError(413, "Die Anfrage ist zu groß; Bilder gehen über /v1/files.")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise RequestError(400, "Der Inhalt ist kein JSON.")
        if not isinstance(body, dict):
            raise RequestError(400, "Der Inhalt ist kein JSON-Objekt.")
        return body

    def app(self) -> str:
        """Who asks: the gate's convention (Bearer <app>), else X-Llm-App, else the address."""
        bearer = re.match(r"Bearer\s+(\S+)", self.headers.get("Authorization") or "", re.I)
        if bearer:
            return bearer[1][:64]
        return (self.headers.get("X-Llm-App") or f"unbekannt@{self.client_address[0]}")[:64]

    def priority(self) -> int:
        try:
            return int(self.headers.get("X-Img-Priority") or DEFAULT_PRIORITY)
        except ValueError:
            return DEFAULT_PRIORITY

    def client_gone(self) -> bool:
        """Whether the client hung up (a phone locked, a cancel): its socket reads as closed."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def route(self) -> tuple[str, dict]:
        parsed = urllib.parse.urlsplit(self.path)
        return parsed.path.rstrip("/") or "/", dict(urllib.parse.parse_qsl(parsed.query))

    def handle_safely(self, work) -> None:
        try:
            work()
        except (RequestError, FileError, LoraError) as error:
            self.send_error_json(error.status, error.message)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # one request must not take the server down
            log(f"{self.command} {self.path}: {error!r}")
            try:
                self.send_error_json(500, str(error))
            except OSError:
                pass

    # --- verbs -----------------------------------------------------------------------------------------------------

    def do_GET(self):
        self.handle_safely(self._get)

    def do_HEAD(self):
        self.handle_safely(self._head)

    def do_POST(self):
        self.handle_safely(self._post)

    def do_PUT(self):
        self.handle_safely(self._put)

    def do_PATCH(self):
        self.handle_safely(self._patch)

    def do_DELETE(self):
        self.handle_safely(self._delete)

    def _get(self):
        path, query = self.route()
        service = self.service
        if path == "/":
            with open(UI, "rb") as file:
                body = file.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/v1/models":
            self.send_json(200, {"object": "list", "data": service.models(fresh=query.get("fresh") == "1")})
        elif path == "/status":
            self.send_json(200, service.status())
        elif path == "/v1/loras":
            self.send_json(200, service.loras.list())
        else:
            self.send_error_json(404, f"Unbekannt: GET {path}")

    def _head(self):
        path, _ = self.route()
        match = re.fullmatch(r"/v1/files/([0-9a-f]{64})", path)
        if match:
            self.send_response(200 if self.service.files.path(match[1]) else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error_json(404, f"Unbekannt: HEAD {path}")

    def _post(self):
        path, _ = self.route()
        service = self.service
        if path == "/v1/images/generations":
            self.generate(self.read_json())
            return
        match = re.fullmatch(r"/v1/models/([^/]+)/download", path)
        if match:
            model = service.catalogue.find(match[1])
            if model is None:
                raise RequestError(404, f"Unbekanntes Modell {match[1]}.")
            probe = service.probe(fresh=True)
            if probe is None:
                raise RequestError(502, service.link.probe_error or "Der Mac, der malt, antwortet nicht.")
            if probe.get("freeGb") is not None and probe["freeGb"] < model["downloadGb"] + float(service.config.get("minFreeGb", 20)):
                raise RequestError(507, f"Auf {probe.get('modelCache')} sind nur {probe['freeGb']} GB frei, {model['label']} braucht {model['downloadGb']} GB.")
            self.send_json(202, service.downloads.start(model))
            return
        if path == "/v1/loras/fetch":
            body = self.read_json()
            suggestion = next((s for s in SUGGESTIONS if s["name"] == body.get("name")), None)
            if suggestion is None:
                raise RequestError(404, f"Keinen Vorschlag {body.get('name')!r}.")
            info = service.loras.fetch(suggestion)
            if info is None:
                self.send_json(202, {"fetching": suggestion["name"]})
            else:
                self.send_json(200, info)
            return
        self.send_error_json(404, f"Unbekannt: POST {path}")

    def _put(self):
        path, query = self.route()
        length = int(self.headers.get("Content-Length") or 0)
        match = re.fullmatch(r"/v1/files/([^/]+)", path)
        if match:
            created = self.service.files.store(match[1], self.rfile, length)
            self.send_json(201 if created else 200, {"sha256": match[1]})
            return
        match = re.fullmatch(r"/v1/loras/([^/]+)", path)
        if match:
            name = urllib.parse.unquote(match[1])
            settings_given = {k: query[k] for k in ("label", "model", "trigger", "strength") if k in query}
            self.send_json(200, self.service.loras.save(name, self.rfile, length, settings_given, query.get("source")))
            return
        self.send_error_json(404, f"Unbekannt: PUT {path}")

    def _patch(self):
        path, _ = self.route()
        match = re.fullmatch(r"/v1/loras/([^/]+)", path)
        if match:
            info = self.service.loras.update(urllib.parse.unquote(match[1]), self.read_json())
            if info is None:
                raise RequestError(404, "Diese LoRA gibt es nicht.")
            self.send_json(200, info)
            return
        self.send_error_json(404, f"Unbekannt: PATCH {path}")

    def _delete(self):
        path, _ = self.route()
        match = re.fullmatch(r"/v1/loras/([^/]+)", path)
        if match:
            if not self.service.loras.delete(urllib.parse.unquote(match[1])):
                raise RequestError(404, "Diese LoRA gibt es nicht.")
            self.send_json(200, {"deleted": True})
            return
        match = re.fullmatch(r"/v1/jobs/([^/]+)", path)
        if match:
            job = self.service.jobs.get(match[1])
            if job is None:
                raise RequestError(404, "Diesen Auftrag gibt es nicht.")
            self.service.painter.cancel(job)
            self.send_json(200, {"cancelled": True})
            return
        self.send_error_json(404, f"Unbekannt: DELETE {path}")

    # --- painting --------------------------------------------------------------------------------------------------

    def generate(self, body: dict) -> None:
        service = self.service
        job = service.job_from(body, self.app(), self.priority())
        service.painter.submit(job)
        if body.get("stream"):
            self.stream(job)
            return
        # OpenAI's plain answer: the pictures when they are done. A client that gives up cancels the picture.
        for event in job.events(timeout=2):
            if event is None and self.client_gone():
                service.painter.cancel(job)
                return
        if job.error:
            self.send_error_json(job.error[0], job.error[1])
        else:
            self.send_json(200, answer_of(job), {"X-Img-Job": job.id})

    def stream(self, job: Job) -> None:
        """Server-sent events: queued (place in line), progress (stage, fraction), image (each picture as it is done),
        then done with the whole answer, or error."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Img-Job", job.id)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(name: str, data) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            send("started", {"id": job.id, "model": job.model["id"], "seed": job.request["seed"]})
            quiet = time.time()
            for event in job.events(timeout=2):
                if event is None:
                    if self.client_gone():
                        raise BrokenPipeError()
                    if time.time() - quiet > PING_SECONDS:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        quiet = time.time()
                    continue
                quiet = time.time()
                kind = event.pop("event")
                if kind == "image":
                    result = event
                    send("image", {"index": result["index"], "b64_json": base64.b64encode(result["jpeg"]).decode("ascii"),
                                   "revised_prompt": result["revisedPrompt"], "seed": result["seed"],
                                   "loadSeconds": result["loadSeconds"], "paintSeconds": result["paintSeconds"],
                                   "peakMemoryGb": result["peakMemoryGb"]})
                else:
                    send(kind, event)
            if job.error:
                send("error", {"error": {"message": job.error[1], "type": "img_o_mat", "code": job.error[0]}})
            else:
                send("done", {"id": job.id, "count": len(job.results)})
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.service.painter.cancel(job)


def main() -> int:
    config = settings.load()
    Handler.service = Service(config)
    server = ThreadingHTTPServer((config.listen, config.port), Handler)
    server.daemon_threads = True
    log(f"img-o-mat hört auf {config.listen}:{config.port}, malt auf {Handler.service.link.name}, Daten in {config.root}.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
