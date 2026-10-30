"""The service's settings: ~/.config/img-o-mat/config.json (IMG_O_MAT_CONFIG elsewhere), every key optional.

    {"listen": "0.0.0.0", "port": 1236, "worker": {"ssh": "marcel@oi-o-mat", "key": null},
     "root": "~/Library/Application Support/img-o-mat",
     "tokenFile": "~/.config/img-o-mat/huggingface.token", "systemMemoryGb": null, "minFreeGb": 20,
     "timeoutMinutes": 90, "keepFilesDays": 7, "models": [...],
     "metrics": {"apps": ["tonwerk-ai", "3d-o-mat", "img-o-mat"]},
     "loki": {"url": "http://monitoring:3100", "host": "proxmox"}}

Where the Python environments and the models lie is the worker's business (worker/common.py, worker.json on
oi-o-mat). The token here is only for fetching LoRAs from Hugging Face; the one on oi-o-mat downloads the models.

It listens on every address, since Tonwerk-AI and the LLM gate on oi-o-mat ask it over the network; like Tonwerk and
the gate it has no login and belongs in the home network and Tailscale only. Apps are meant to come through the gate,
which decides who may use which model. Without "worker" it paints on its own Mac.
"""

from __future__ import annotations

import json
import os
import sys

HOME = os.path.expanduser("~")


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def default_root() -> str:
    if sys.platform == "darwin":
        return os.path.join(HOME, "Library", "Application Support", "img-o-mat")
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(HOME, ".local", "share"), "img-o-mat")


class Config(dict):
    def __init__(self, values: dict):
        super().__init__(values)
        self.root = expand(self.get("root") or default_root())
        self.files = os.path.join(self.root, "files")
        self.work = os.path.join(self.root, "work")
        self.listen = self.get("listen") or "0.0.0.0"
        self.port = int(self.get("port") or 1236)
        self.timeout = float(self.get("timeoutMinutes") or 90) * 60
        self.keep_files = float(self.get("keepFilesDays") or 7) * 86400

    def token_files(self) -> list[str]:
        if self.get("tokenFile"):
            return [expand(self["tokenFile"])]
        # Tonwerk-AI's place from before the service, so the token need not move.
        return [expand("~/.config/img-o-mat/huggingface.token"), expand("~/.config/tonwerk/huggingface.token")]

    def token(self) -> str | None:
        """Read for every picture, so a token put there later works without a restart."""
        for path in self.token_files():
            try:
                with open(path) as file:
                    if token := file.read().strip():
                        return token
            except OSError:
                continue
        return None


def load() -> Config:
    path = expand(os.environ.get("IMG_O_MAT_CONFIG") or "~/.config/img-o-mat/config.json")
    try:
        with open(path) as file:
            return Config(json.load(file))
    except FileNotFoundError:
        return Config({})
