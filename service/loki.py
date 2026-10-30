"""Sends the service's log lines to Loki as well (config "loki": {"url": "http://monitoring:3100", "host": "proxmox"};
no url means off). stdout and with it the journal stay as they are. Labels only app, host, level and source.

A thread collects lines and pushes them every few seconds. When Loki is away, lines are dropped rather than holding
the service up: at most MAX_LINES wait, and a failed push is counted, never logged, since that line would be the next
one to fail.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
import urllib.request

MAX_LINES = 5000


class LokiShipper:
    def __init__(self, settings: dict | None, sent, dropped, interval: float = 5, timeout: float = 10):
        settings = settings or {}
        url = str(settings.get("url") or "").rstrip("/")
        self.url = f"{url}/loki/api/v1/push" if url else None
        self.host = settings.get("host") or socket.gethostname().split(".")[0]
        self.interval = float(settings.get("intervalSeconds") or interval)
        self.timeout = timeout
        self.sent = sent
        self.dropped = dropped
        self._lines: queue.Queue = queue.Queue(maxsize=MAX_LINES)
        self._last_ns = 0
        self._lock = threading.Lock()
        if self.url:
            threading.Thread(target=self._run, daemon=True, name="loki").start()

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def add(self, message: str, level: str, source: str) -> None:
        if not self.url:
            return
        with self._lock:
            # Loki wants nanoseconds as text and keeps only one line per stream and time: lines in the same
            # nanosecond move one up.
            ns = max(time.time_ns(), self._last_ns + 1)
            self._last_ns = ns
        try:
            self._lines.put_nowait((str(ns), message, level, source))
        except queue.Full:
            self.dropped.inc()

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            self.flush()

    def flush(self) -> None:
        batch = []
        while True:
            try:
                batch.append(self._lines.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return
        streams: dict[tuple, dict] = {}
        for ns, message, level, source in batch:
            stream = streams.setdefault((level, source), {
                "stream": {"app": "img-o-mat", "host": self.host, "level": level, "source": source}, "values": []})
            stream["values"].append([ns, message])
        body = json.dumps({"streams": list(streams.values())}).encode()
        request = urllib.request.Request(self.url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            self.sent.inc(len(batch))
        except Exception:  # any failure: the batch is gone, counted
            self.dropped.inc(len(batch))
