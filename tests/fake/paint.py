"""Stands in for worker/paint.py in the tests: no model, a tiny JPEG, the same events. FAKE in the prompt steers it:
"fail" ends with an error line, "slow" paints for a minute (to be cancelled), "filter" is held back by the filter."""

import base64
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("IMG_O_MAT_SOURCE_WORKER") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "worker"))
import common  # noqa: E402

# 1x1 white JPEG.
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////"
    "////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)

request = json.load(sys.stdin)
with open(os.path.join(common.ROOT, "last-request.json"), "w") as out:
    json.dump(request, out)
for item in (request.get("loras") or []) + (request.get("references") or []):
    if not os.path.exists(os.path.join(common.settings()["files"], item["file"])):
        print(f"Error: file {item['file']} missing", flush=True)
        sys.exit(1)
prompt = request["prompt"]
common.emit({"event": "stage", "stage": "loading", "fraction": 0})
print("mflux says hello", flush=True)
if "slow" in prompt:
    with open(os.path.join(common.ROOT, "slow.pid"), "w") as out:
        out.write(str(os.getpid()))
    time.sleep(60)
if "fail" in prompt:
    print("Traceback ...\nError: the model broke", flush=True)
    sys.exit(1)
if "filter" in prompt and request["contentFilter"]:
    print("Error: the content filter held the picture back; describe it differently.", flush=True)
    sys.exit(1)
common.emit({"event": "stage", "stage": "painting", "fraction": 0.5})
common.emit({"event": "done", "jpeg": base64.b64encode(JPEG).decode(), "loadSeconds": 0.1, "paintSeconds": 0.2,
             "peakMemoryGb": 1.0, "paintedPrompt": "upsampled " + prompt if request.get("upsampler") else None})
