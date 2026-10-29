"""Tells the service what this Mac has: its memory, which runners are installed, which repositories are in the cache,
whether a Hugging Face token is there, which of the given files are here and how much the cache's volume has free.
Plain Python 3.9 and nothing else, since the service asks it often.

In: {"repos": ["black-forest-labs/FLUX.2-klein-4B", ...], "files": ["<sha256>", ...]} on stdin
Out: one JSON object on stdout.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def memory_gb() -> float | None:
    """In GiB as Apple counts it (a 32 GB Mac says 32)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / (1 << 30)
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1 << 20)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


KEEP_FILES_DAYS = 30


def prune(folder: str) -> None:
    """LoRAs and reference pictures the service sent and nobody used for a month; it sends them again if needed."""
    limit = time.time() - KEEP_FILES_DAYS * 86400
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        try:
            path = os.path.join(folder, name)
            if os.path.getmtime(path) < limit:
                os.remove(path)
        except OSError:
            pass


def main() -> int:
    asked = json.load(sys.stdin)
    values = common.settings()
    prune(values["files"])
    hub = os.path.join(values["modelCache"], "hub")
    cache = values["modelCache"] if os.path.exists(values["modelCache"]) else common.ROOT
    try:
        free = shutil.disk_usage(cache).free / 1e9
    except OSError:
        free = None
    json.dump(
        {
            "host": platform.node(),
            "memoryGb": memory_gb(),
            "runners": {"mflux": os.path.exists(values["python"]), "flux3-mlx": os.path.exists(values["flux3Python"])},
            "downloaded": {
                repo: os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots"))
                for repo in asked.get("repos") or []
            },
            "tokenFound": common.token(values) is not None,
            "files": [d for d in asked.get("files") or [] if os.path.exists(os.path.join(values["files"], d))],
            "modelCache": values["modelCache"],
            "freeGb": round(free, 1) if free is not None else None,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
