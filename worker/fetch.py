"""Downloads a model's repositories into the cache, for the service's download button (and the LLM gate's
"Installieren"). Runs with mflux's Python, which has huggingface_hub; the token is read like paint.py does.

In: {"repos": ["krea/Krea-2-Turbo", ...], "expectedGb": 35} on stdin
Out: "IMGOMAT " events: {"event": "progress", "gb": 3.2, "fraction": 0.09} every few seconds, then {"event": "done"};
on failure an "Error: ..." line and exit code 1.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def size_gb(folder: str) -> float:
    total = 0
    for base, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total / 1e9


def main() -> int:
    asked = json.load(sys.stdin)
    values = common.settings()
    common.prepare_environment(values)
    hub = os.path.join(values["modelCache"], "hub")
    os.makedirs(hub, exist_ok=True)
    folders = [os.path.join(hub, "models--" + repo.replace("/", "--")) for repo in asked["repos"]]
    # Downloads land in shared blobs on Marcel's T7 too (hub/blobs), so the whole hub is measured against the start.
    start = size_gb(hub)
    expected = float(asked.get("expectedGb") or 0)
    finished = threading.Event()

    def report() -> None:
        while not finished.wait(5):
            got = max(size_gb(hub) - start, 0)
            common.emit({"event": "progress", "gb": round(got, 1), "fraction": round(min(got / expected, 0.99), 3) if expected else None})

    threading.Thread(target=report, daemon=True).start()
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("Error: huggingface_hub fehlt; install.sh --worker noch einmal ausführen.", flush=True)
        return 1
    try:
        for repo in asked["repos"]:
            snapshot_download(repo)
    except Exception as error:
        name = type(error).__name__
        if name in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(error) or "403" in str(error):
            print(
                f"Error: Hugging Face verweigert {repo}. Die Lizenz auf huggingface.co annehmen und einen Token dieses"
                " Kontos in ~/.config/tonwerk/huggingface.token legen.",
                flush=True,
            )
        else:
            print(f"Error: {name}: {error}", flush=True)
        return 1
    finally:
        finished.set()
    common.emit({"event": "done", "gb": round(sum(size_gb(f) for f in folders), 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
