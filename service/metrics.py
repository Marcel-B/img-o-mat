"""Measurements for Prometheus (GET /metrics), written by hand in its text format so the service stays without
packages. Counters and histograms are kept here, thread-safe; what is a state right now (the queue, the worker) the
server reads when it is asked. Conventions as the other apps' (scripts repository, docs/monitoring.md): the name starts
with imgomat_, counters end in _total, durations in _seconds, labels only take a few fixed values.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

# A picture takes seconds (Klein 4B, loaded) to many minutes (FLUX 3 with its upsampler, the first after a download).
PAINT_BUCKETS = (5, 10, 20, 30, 60, 120, 300, 600)
# Waiting runs from nothing to the length of a few pictures before it.
WAIT_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1800)

START = time.time()


def _escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(labels: dict) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in labels.items()) + "}"


def _number(value: float) -> str:
    if value == float("inf"):
        return "+Inf"
    return repr(float(value)) if isinstance(value, float) and not value.is_integer() else str(int(value))


def _key(labels: dict) -> tuple:
    return tuple(sorted(labels.items()))


class Counter:
    def __init__(self, name: str, help: str):
        self.name, self.help = name, help
        self._values: dict[tuple, list] = {}
        self._lock = threading.Lock()

    def inc(self, by: float = 1, **labels) -> None:
        with self._lock:
            entry = self._values.setdefault(_key(labels), [dict(labels), 0])
            entry[1] += by

    def value(self, **labels) -> float:
        with self._lock:
            entry = self._values.get(_key(labels))
            return entry[1] if entry else 0

    def write(self, out: list) -> None:
        out += [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for labels, value in self._values.values():
                out.append(f"{self.name}{_labels(labels)} {_number(value)}")


class Histogram:
    def __init__(self, name: str, help: str, buckets: tuple):
        self.name, self.help, self.buckets = name, help, buckets
        self._series: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        with self._lock:
            series = self._series.setdefault(
                _key(labels), {"labels": dict(labels), "counts": [0] * len(self.buckets), "sum": 0.0, "count": 0})
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    series["counts"][index] += 1
            series["sum"] += value
            series["count"] += 1

    def write(self, out: list) -> None:
        out += [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for series in self._series.values():
                labels = series["labels"]
                for bound, count in zip(self.buckets, series["counts"]):
                    out.append(f"{self.name}_bucket{_labels({**labels, 'le': _number(bound)})} {count}")
                out.append(f"{self.name}_bucket{_labels({**labels, 'le': '+Inf'})} {series['count']}")
                out.append(f"{self.name}_sum{_labels(labels)} {_number(series['sum'])}")
                out.append(f"{self.name}_count{_labels(labels)} {series['count']}")


def gauge(out: list, name: str, help: str, samples) -> None:
    """A state read at the scrape: samples are (labels, value) pairs; none still writes the family's header."""
    out += [f"# HELP {name} {help}", f"# TYPE {name} gauge"]
    for labels, value in samples:
        out.append(f"{name}{_labels(labels)} {_number(value)}")


def process(out: list) -> None:
    """The process' memory, CPU time and start, the names every exporter uses."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    resident = None
    try:
        with open("/proc/self/statm") as file:
            resident = int(file.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        # macOS (the service on a Mac): only the peak, in bytes there, kilobytes on Linux.
        resident = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    gauge(out, "process_resident_memory_bytes", "Resident memory of the service.", [({}, resident)])
    out += ["# HELP process_cpu_seconds_total CPU time of the service.", "# TYPE process_cpu_seconds_total counter",
            f"process_cpu_seconds_total {_number(usage.ru_utime + usage.ru_stime)}"]
    gauge(out, "process_start_time_seconds", "When the service started, Unix time.", [({}, START)])


def folder_bytes(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return total


class Metrics:
    """Everything the service counts. One instance, kept by the Service."""

    def __init__(self, apps):
        # Only these names become an app label; everything else is "other", so a key nobody knows cannot add series.
        self.apps = set(apps)
        self.images = Counter("imgomat_images_total", "Pictures by app, model and result (done, failed, filtered, cancelled).")
        self.paint = Histogram("imgomat_paint_seconds", "Time one picture took, from starting the runner to its answer.", PAINT_BUCKETS)
        self.wait = Histogram("imgomat_wait_seconds", "Time a job waited in the queue before it started.", WAIT_BUCKETS)
        self.worker_errors = Counter("imgomat_worker_errors_total", "Failures reaching the Mac that paints, by kind (ssh, runner, transfer).")
        self.downloads = Counter("imgomat_downloads_total", "Model downloads by model and result (done, failed).")
        self.requests = Counter("imgomat_http_requests_total", "HTTP requests by method, route pattern and status.")
        self.loki_sent = Counter("imgomat_loki_lines_sent_total", "Log lines Loki took.")
        self.loki_dropped = Counter("imgomat_loki_dropped_total", "Log lines dropped because Loki did not take them or the buffer was full.")

    def app(self, name: str) -> str:
        return name if name in self.apps else "other"

    def render(self, states) -> str:
        """states(out) writes the gauges read now (queue, painting, worker, files)."""
        out: list[str] = []
        for counter in (self.images, self.worker_errors, self.downloads, self.requests, self.loki_sent, self.loki_dropped):
            counter.write(out)
        self.paint.write(out)
        self.wait.write(out)
        states(out)
        process(out)
        return "\n".join(out) + "\n"
