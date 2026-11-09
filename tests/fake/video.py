"""Stands in for worker/video.py in the tests: no model, a few bytes that start like an mp4, the same events. "fail" in
the prompt ends with an error line."""

import base64
import json
import os
import sys

sys.path.insert(0, os.environ.get("IMG_O_MAT_SOURCE_WORKER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "worker"))
import common  # noqa: E402

request = json.load(sys.stdin)
with open(os.path.join(common.ROOT, "last-video.json"), "w") as out:
    json.dump(request, out)
for image in request.get("images") or []:
    if not os.path.exists(os.path.join(common.settings()["files"], image["file"])):
        print(f"Error: Bild {image['file'][:12]} fehlt auf dem Mac.", flush=True)
        sys.exit(1)
common.emit({"event": "stage", "stage": "loading", "fraction": 0})
print("ltx-2-mlx says hello", flush=True)
if "fail" in request["prompt"]:
    print("Error: ltx-2-mlx endete mit 1 ohne Video.", flush=True)
    sys.exit(1)
common.emit({"event": "stage", "stage": "painting", "fraction": 0.5})
common.emit({"event": "done", "mp4": base64.b64encode(b"\0\0\0\x18ftypisom fake clip").decode(), "loadSeconds": None,
             "paintSeconds": 1.5, "peakMemoryGb": None})
