"""Stands in for worker/fetch.py: makes the repositories' snapshot folders in the cache."""

import json
import os
import sys

sys.path.insert(0, os.environ.get("IMG_O_MAT_SOURCE_WORKER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "worker"))
import common  # noqa: E402

asked = json.load(sys.stdin)
hub = os.path.join(common.settings()["modelCache"], "hub")
for repo in asked["repos"]:
    os.makedirs(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots"), exist_ok=True)
    common.emit({"event": "progress", "gb": 1, "fraction": 0.5})
common.emit({"event": "done", "gb": 2})
