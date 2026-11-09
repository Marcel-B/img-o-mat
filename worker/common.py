"""What the worker scripts share: where the worker lives on the Mac that paints (oi-o-mat), its settings and the
Hugging Face token. install.sh --worker copies this folder to <root>/worker/ and writes <root>/worker.json:

    {"modelCache": "/Volumes/T7/...", "tokenFile": "~/.config/tonwerk/huggingface.token",
     "python": "<root>/env/bin/python", "flux3Python": "<root>/flux3/bin/python", "ltxPython": "<root>/ltx/bin/python",
     "gallery": "/Volumes/T7/..."}

Without "gallery" the kept pictures go on the disk the models are on (img-o-mat/galerie at its top), else <root>/gallery.

The scripts run with the runner's Python (paint.py, fetch.py) or any Python 3.9 (probe.py), started by the service over
SSH or, where the service paints on its own Mac, as plain child processes.
"""

from __future__ import annotations

import json
import os

# What the service reads as an event; everything else on stdout is the runner talking.
PREFIX = "IMGOMAT "

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("IMG_O_MAT_WORKER_ROOT") or os.path.dirname(HERE)


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def volume_of(path: str) -> str | None:
    """The external disk a path lies on (/Volumes/<name>/...), followed through links: on oi-o-mat the model cache is
    a link from the Library to the T7. None for the Mac's own disk."""
    parts = os.path.realpath(path).split(os.sep)
    return parts[2] if len(parts) > 3 and parts[1] == "Volumes" else None


def settings() -> dict:
    try:
        with open(os.path.join(ROOT, "worker.json")) as file:
            values = json.load(file)
    except FileNotFoundError:
        values = {}
    cache = expand(values.get("modelCache") or os.path.join(ROOT, "models"))
    disk = volume_of(cache)
    gallery = expand(values["gallery"]) if values.get("gallery") else (
        os.path.join("/Volumes", disk, "img-o-mat", "galerie") if disk else os.path.join(ROOT, "gallery"))
    return {
        "modelCache": cache,
        "tokenFiles": [expand(values["tokenFile"])] if values.get("tokenFile") else [
            expand("~/.config/img-o-mat/huggingface.token"),
            # Tonwerk-AI's place from before the service, so the token need not move.
            expand("~/.config/tonwerk/huggingface.token"),
        ],
        "python": expand(values.get("python") or os.path.join(ROOT, "env", "bin", "python")),
        "flux3Python": expand(values.get("flux3Python") or os.path.join(ROOT, "flux3", "bin", "python")),
        "ltxPython": expand(values.get("ltxPython") or os.path.join(ROOT, "ltx", "bin", "python")),
        "files": expand(values.get("files") or os.path.join(ROOT, "files")),
        "gallery": gallery,
        "galleryDisk": volume_of(gallery),
        # Tests put fake paint.py and fetch.py here.
        "scripts": expand(values.get("scripts") or HERE),
    }


def token(values: dict) -> str | None:
    """Read for every picture, so a token put there later works without a restart."""
    for path in values["tokenFiles"]:
        try:
            with open(path) as file:
                if found := file.read().strip():
                    return found
        except OSError:
            continue
    return None


def prepare_environment(values: dict) -> None:
    """Hugging Face's cache and token for this process, before anything imports huggingface_hub."""
    os.environ["HF_HOME"] = values["modelCache"]
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    if found := token(values):
        os.environ["HF_TOKEN"] = found


def emit(event: dict) -> None:
    print(PREFIX + json.dumps(event), flush=True)
