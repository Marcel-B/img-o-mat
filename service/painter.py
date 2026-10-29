"""One picture at a time, each in its own process (worker/paint.py with the runner's Python, over SSH on the Mac
that paints), so the model and MLX's Metal cache leave the memory with the process. Waiting jobs go by priority, then by arrival: Tonwerk-AI asks with priority 0
and goes before what other apps send through the gate."""

from __future__ import annotations

import base64
import itertools
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque

EVENT_PREFIX = "IMGOMAT "
TAIL_LINES = 30


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, model: dict, request: dict, app: str, priority: int):
        self.id = uuid.uuid4().hex[:12]
        self.model = model
        self.request = request
        self.app = app
        self.priority = priority
        self.created = time.time()
        self.started: float | None = None
        self.stage = "queued"
        self.fraction = 0.0
        self.results: list[dict] = []
        self.error: tuple[int, str] | None = None
        self.done = False
        self.cancelled = False
        self.process = None
        self._events: list[dict] = []
        self._changed = threading.Condition()

    def emit(self, event: dict) -> None:
        with self._changed:
            self._events.append(event)
            self._changed.notify_all()

    def events(self, timeout: float):
        """Yields events as they come; None after `timeout` seconds without one, so the caller can look at its client."""
        seen = 0
        while True:
            with self._changed:
                if seen >= len(self._events) and not self.done:
                    self._changed.wait(timeout)
                fresh = self._events[seen:]
                seen = len(self._events)
                finished = self.done
            if not fresh:
                if finished:
                    return
                yield None
            yield from fresh
            if finished and seen >= len(self._events):
                return

    def summary(self) -> dict:
        return {
            "id": self.id,
            "app": self.app,
            "model": self.model["id"],
            "priority": self.priority,
            "stage": self.stage,
            "fraction": self.fraction,
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started or self.created)),
        }


class Painter:
    def __init__(self, config, link, log):
        self.config = config
        self.link = link
        self.log = log
        self.queue: list[Job] = []
        self.running: Job | None = None
        self.finished = deque(maxlen=20)
        self._order = itertools.count()
        self._ranks: dict[str, int] = {}
        self._wake = threading.Condition()
        threading.Thread(target=self._work, daemon=True, name="painter").start()

    def submit(self, job: Job) -> None:
        with self._wake:
            self._ranks[job.id] = next(self._order)
            self.queue.append(job)
            self.queue.sort(key=lambda j: (j.priority, self._ranks[j.id]))
            self._announce_places()
            self._wake.notify_all()

    def cancel(self, job: Job) -> None:
        with self._wake:
            job.cancelled = True
            if job in self.queue:
                self.queue.remove(job)
                self._finish(job, (499, "abgebrochen"))
                self._announce_places()
        if job.process and job.process.poll() is None:
            self._stop(job.process)

    def status(self) -> dict:
        with self._wake:
            return {
                "running": self.running.summary() if self.running else None,
                "queue": [job.summary() for job in self.queue],
                "recent": list(self.finished),
            }

    def _announce_places(self) -> None:
        for place, job in enumerate(self.queue, start=1):
            job.emit({"event": "queued", "place": place, "running": self.running.app if self.running else None})

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self.queue:
                    self._wake.wait()
                job = self.queue.pop(0)
                self.running = job
                self._announce_places()
            job.started = time.time()
            try:
                self._paint(job)
                self._finish(job, None)
            except Cancelled:
                self._finish(job, (499, "abgebrochen"))
            except PaintError as error:
                self.log(f"{job.app}: {job.model['id']} fehlgeschlagen: {error.message}")
                self._finish(job, (error.status, error.message))
            except Exception as error:  # the queue must keep going whatever one picture does
                self.log(f"{job.app}: {job.model['id']} fehlgeschlagen: {error!r}")
                self._finish(job, (500, str(error)))
            finally:
                with self._wake:
                    self.running = None

    def _finish(self, job: Job, error: tuple[int, str] | None) -> None:
        job.error = error
        job.stage = "failed" if error else "done"
        self.finished.appendleft({**job.summary(), "error": error[1] if error else None})
        self._ranks.pop(job.id, None)
        with job._changed:
            job.done = True
            job._changed.notify_all()

    def _paint(self, job: Job) -> None:
        count = int(job.request.get("n") or 1)
        for index in range(count):
            if job.cancelled:
                raise Cancelled()
            result = self._paint_one(job, int(job.request["seed"]) + index, index, count)
            job.results.append(result)
            job.emit({"event": "image", "index": index, **result})

    def _paint_one(self, job: Job, seed: int, index: int, count: int) -> dict:
        model = job.model
        job.stage = "sending"
        for digest, path in (job.request.get("files") or {}).items():
            try:
                self.link.send_file(digest, path)
            except Exception as error:
                raise PaintError(502, str(error))
            if job.cancelled:
                raise Cancelled()
        request = {
            "runner": model["runner"],
            "model": model["model"],
            "label": model["label"],
            "creator": job.request.get("creator") or job.app,
            "quantize": model["quantize"],
            "steps": int(job.request.get("steps") or model["steps"]),
            "upsampler": model["upsampler"],
            # Always the catalogue's: the licences ask for the filter, no client may leave it out.
            "contentFilter": model["contentFilter"],
            "prompt": job.request["prompt"],
            "seed": seed,
            "width": job.request["width"],
            "height": job.request["height"],
            "loras": job.request.get("loras") or [],
            "references": job.request.get("references") or [],
        }
        self.log(f"{job.app}: male mit {model['id']} auf {self.link.name} ({index + 1}/{count}).")
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        done: dict | None = None
        try:
            process = self.link.start("paint", model["runner"])
        except OSError as error:
            raise PaintError(500, f"Malen startet nicht: {error}")
        job.process = process
        timer = threading.Timer(self.config.timeout, self._stop, [process])
        timer.start()
        started = time.time()
        try:
            process.stdin.write((json.dumps(request) + "\n").encode())
            process.stdin.flush()
            for raw in process.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line.startswith(EVENT_PREFIX):
                    tail.append(line)
                    continue
                data = json.loads(line[len(EVENT_PREFIX) :])
                if data.get("event") == "stage":
                    job.stage = data.get("stage") or "painting"
                    # Across n pictures the share of the whole job.
                    job.fraction = round((index + float(data.get("fraction") or 0)) / count, 3)
                    job.emit({"event": "progress", "stage": job.stage, "fraction": job.fraction})
                elif data.get("event") == "done":
                    done = data
            process.wait()
        finally:
            timer.cancel()
            job.process = None
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        if job.cancelled:
            raise Cancelled()
        if process.returncode != 0 or done is None or not done.get("jpeg"):
            if process.returncode and process.returncode < 0 and time.time() - started >= self.config.timeout:
                raise PaintError(504, f"Das Modell brauchte länger als {self.config.timeout / 60:.0f} Minuten.")
            if process.returncode == 255 and self.link.ssh and not done:
                # ssh's own exit code: the connection, not the picture, failed.
                tail.append(f"Error: {self.link.name} nicht erreichbar über SSH.")
            self.log("\n".join(tail))
            raise PaintError(filter_status(tail), failure_message(tail))
        return {
            "jpeg": base64.b64decode(done["jpeg"]),
            "seed": seed,
            "revisedPrompt": done.get("paintedPrompt"),
            "loadSeconds": done.get("loadSeconds"),
            "paintSeconds": done.get("paintSeconds"),
            "peakMemoryGb": done.get("peakMemoryGb"),
        }

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        """Closes the runner's stdin, which launch.py on the other end takes as the signal to kill the picture (over
        SSH nothing else reaches it); whatever still runs after a few seconds is killed here."""
        try:
            process.stdin.close()
        except OSError:
            pass

        def kill() -> None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        threading.Timer(5, kill).start()


class PaintError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def failure_message(output) -> str:
    """paint.py prints its own "Error: …" line where it knows better (a gated repository, a missing package, the
    content filter); else the traceback's last line says what broke."""
    lines = [line.strip() for line in output if line.strip()]
    line = next((l for l in reversed(lines) if l.startswith("Error")), lines[-1] if lines else None)
    return line.removeprefix("Error: ") if line else "Das Modell hat kein Bild gemalt."


def filter_status(output) -> int:
    """A picture the content filter held back is the request's fault (422), not the service's."""
    return 422 if any("content filter" in line for line in output) else 500
