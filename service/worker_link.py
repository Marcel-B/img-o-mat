"""How the service reaches the Mac that paints: over SSH (config "worker": {"ssh": "marcel@oi-o-mat"}), or on
its own Mac when there is no "worker" (then the scripts run from this checkout and the service's root is the worker's
too). Everything goes through worker/launch.py, so both ways run the same commands.

SSH runs non-interactively (BatchMode) with a key oi-o-mat allows only worker/launch.py (deploy/to-proxmox.sh puts it
there); "ssh" may also name a host of ~/.ssh/config, "key" a key file other than the user's default.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_WORKER = os.path.join(os.path.dirname(HERE), "worker")

# The probe is cheap but crosses the network; listings use the last one unless it is older than this.
PROBE_SECONDS = 30


class WorkerError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class WorkerLink:
    def __init__(self, config, log, metrics=None):
        self.config = config
        self.log = log
        self.metrics = metrics
        worker = config.get("worker") or {}
        self.ssh = worker.get("ssh")
        self.key = worker.get("key")
        self._probe: dict | None = None
        self._probed_at = 0.0
        self._probe_error: str | None = None
        self._ever_probed = False
        self._lock = threading.Lock()
        # Hashes the worker is known to have and since when, so a picture does not ask before every LoRA. The worker
        # forgets files unused for a month (probe.py), so what is known here is asked again after a day.
        self._present: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self.ssh or "lokal"

    def command(self, *arguments: str) -> list[str]:
        if not self.ssh:
            return ["/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable,
                    os.path.join(LOCAL_WORKER, "launch.py"), *arguments]
        # The key may only run launch.py (its forced command); these words are what it reads.
        remote = "img-o-mat " + " ".join(shlex.quote(a) for a in arguments)
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            # The first connection learns oi-o-mat's host key; a changed one later is refused.
            "-o", "StrictHostKeyChecking=accept-new",
            # A dead connection ends the picture within a minute instead of the timeout's 90.
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
            *(["-i", os.path.expanduser(self.key)] if self.key else []),
            self.ssh,
            remote,
        ]

    def environment(self) -> dict:
        environment = dict(os.environ)
        if not self.ssh:
            environment["IMG_O_MAT_WORKER_ROOT"] = self.config.root
        return environment

    def start(self, *arguments: str) -> subprocess.Popen:
        """A script with its request to follow on stdin as one line; stdin stays open until the caller closes it."""
        return subprocess.Popen(
            self.command(*arguments),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self.environment(),
            start_new_session=True,
        )

    def probe(self, repos: list[str], fresh: bool = False) -> dict | None:
        """What the worker has, at most PROBE_SECONDS old; None (and probe_error) when it does not answer."""
        with self._lock:
            if not fresh and self._probe is not None and time.time() - self._probed_at < PROBE_SECONDS:
                return self._probe
        try:
            out = subprocess.run(
                self.command("probe"),
                input=json.dumps({"repos": repos}).encode(),
                capture_output=True,
                timeout=30,
                env=self.environment(),
            )
            if out.returncode != 0:
                lines = (out.stderr or out.stdout).decode(errors="replace").strip().splitlines()
                raise WorkerError(lines[-1] if lines else f"Exit {out.returncode}")
            probe = json.loads(out.stdout)
            error = None
        except (OSError, subprocess.SubprocessError, ValueError, WorkerError) as failure:
            probe = None
            message = failure.message if isinstance(failure, WorkerError) else str(failure)
            error = f"{self.name} antwortet nicht: {message}"
        if probe is None and self.metrics:
            self.metrics.worker_errors.inc(kind="ssh" if self.ssh else "runner")
        with self._lock:
            if probe is None and error != self._probe_error:
                self.log(error, "warning")
            elif probe is not None and self._probe_error is not None:
                self.log(f"{self.name} antwortet wieder.")
            self._probe, self._probed_at, self._probe_error = probe, time.time(), error
            self._ever_probed = True
            return probe

    @property
    def probe_error(self) -> str | None:
        return self._probe_error

    @property
    def probed(self) -> bool:
        """Whether the worker was asked at all since the start."""
        return self._ever_probed

    def forget_probe(self) -> None:
        with self._lock:
            self._probed_at = 0

    def send_file(self, digest: str, path: str) -> None:
        """Copies a LoRA or reference picture to the worker unless it is there already."""
        if time.time() - self._present.get(digest, 0) < 86400:
            return
        if not self.ssh:
            # The worker's files folder is the service's own; a library LoRA is linked into it.
            target = os.path.join(self.config.files, digest)
            if not os.path.exists(target):
                os.makedirs(self.config.files, exist_ok=True)
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
            self._present[digest] = time.time()
            return
        probe = subprocess.run(
            self.command("probe"), input=json.dumps({"files": [digest]}).encode(), capture_output=True,
            timeout=30, env=self.environment(),
        )
        if probe.returncode == 0 and digest in json.loads(probe.stdout).get("files", []):
            self._present[digest] = time.time()
            return
        with open(path, "rb") as source:
            sent = subprocess.run(self.command("receive", digest), stdin=source, capture_output=True, timeout=1800)
        if sent.returncode != 0:
            raise WorkerError(f"Datei an {self.name} nicht übertragen: {sent.stderr.decode(errors='replace').strip()}")
        self._present[digest] = time.time()
