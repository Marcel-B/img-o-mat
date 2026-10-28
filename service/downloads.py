"""Model downloads on the Mac that paints (worker/fetch.py), started by the page's button or the LLM gate's
"Installieren". Never by themselves: a model is 15 to 35 GB on the T7. One at a time, the rest wait."""

from __future__ import annotations

import json
import threading
import time

from catalogue import repos

PREFIX = "IMGOMAT "


class Downloads:
    def __init__(self, link, log):
        self.link = link
        self.log = log
        self.states: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._line = threading.Semaphore(1)

    def state(self, model_id: str) -> dict | None:
        with self._lock:
            state = self.states.get(model_id)
            return dict(state) if state else None

    def start(self, model: dict) -> dict:
        with self._lock:
            current = self.states.get(model["id"])
            if current and current["state"] in ("waiting", "running"):
                return dict(current)
            self.states[model["id"]] = {"state": "waiting", "gb": 0, "fraction": None, "error": None,
                                         "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        threading.Thread(target=self._run, args=(model,), daemon=True, name=f"download {model['id']}").start()
        return self.state(model["id"])

    def _set(self, model_id: str, **values) -> None:
        with self._lock:
            self.states[model_id].update(values)

    def _run(self, model: dict) -> None:
        with self._line:
            self._set(model["id"], state="running")
            self.log(f"Lade {model['id']} auf {self.link.name} herunter ({model['downloadGb']} GB).")
            tail = []
            try:
                process = self.link.start("fetch")
                process.stdin.write((json.dumps({"repos": repos(model), "expectedGb": model["downloadGb"]}) + "\n").encode())
                process.stdin.flush()
                for raw in process.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if line.startswith(PREFIX):
                        event = json.loads(line[len(PREFIX) :])
                        if event.get("event") == "progress":
                            self._set(model["id"], gb=event.get("gb"), fraction=event.get("fraction"))
                    elif line.strip():
                        tail = (tail + [line])[-10:]
                process.wait()
                process.stdin.close()
                if process.returncode != 0:
                    error = next((l for l in reversed(tail) if l.startswith("Error")), tail[-1] if tail else "Download fehlgeschlagen")
                    self._set(model["id"], state="failed", error=error.removeprefix("Error: "))
                    self.log(f"Download von {model['id']} fehlgeschlagen: {error}")
                else:
                    self._set(model["id"], state="done", fraction=1)
                    self.log(f"{model['id']} ist heruntergeladen.")
            except Exception as error:  # the thread must leave a state behind
                self._set(model["id"], state="failed", error=str(error))
            finally:
                self.link.forget_probe()
