"""Makes one video clip for img-o-mat with ltx-2-mlx (LTX-2.3). Run with the ltx environment's Python
(install-worker.sh); the service starts it like paint.py, one clip per process, over SSH on the Mac that paints.

ltx-2-mlx runs as its own command line (`ltx-2-mlx generate`), from the local snapshots only: the models came onto the
T7 through the page's download button, and nothing is downloaded here (HF_HUB_OFFLINE). Its progress bars (tqdm on
stderr) become stage events.

In: one JSON object on stdin
    {"runner": "ltx-2-mlx", "model": "dgrauet/ltx-2.3-mlx-q8", "textEncoder": "mlx-community/gemma-3-12b-it-4bit",
     "label": "LTX-2.3 (8 bit)", "creator": "img-o-mat", "prompt": "...", "seed": 42, "width": 768, "height": 768,
     "frames": 97, "fps": 24, "quality": "fast" | "high", "audio": true,
     "images": [{"file": "<sha256>", "at": "first" | "last", "strength": 1.0}]}
    "first" pins a picture as the clip's first frame, "last" as its last: the clip then moves towards it.
Out: "IMGOMAT " events like paint.py's, ending with
    {"event": "done", "mp4": "<base64>", "loadSeconds": null, "paintSeconds": 812.4, "peakMemoryGb": null}
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import emit  # noqa: E402

# Over SSH the PATH is macOS's bare one; ltx-2-mlx needs Homebrew's ffmpeg to write the mp4.
EXTRA_PATH = ["/opt/homebrew/bin", "/usr/local/bin"]

# Where the denoising bars sit in the whole clip: stage 1 at half size, stage 2 refines at full size, the rest is
# loading and decoding.
BARS = [(0.05, 0.75), (0.75, 0.95)]
PROGRESS = re.compile(r"(\d+)/(\d+) \[")


def snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        print(f"Error: {repo} ist nicht auf der Platte; auf der Seite unter Modelle → Videos herunterladen.", flush=True)
        raise SystemExit(1)


def memory_gb() -> float:
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / (1 << 30)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def command(request: dict, values: dict, output: str) -> list[str]:
    files = values["files"]
    work = os.path.dirname(output)
    tool = os.path.join(os.path.dirname(sys.executable), "ltx-2-mlx")
    arguments = [
        tool, "generate",
        "--model", snapshot(request["model"]),
        "--gemma", snapshot(request["textEncoder"]),
        # Distilled: 8 + 3 steps without guidance, the fast way; two-stage: 30 steps with guidance, better and slower.
        "--two-stage" if request.get("quality") == "high" else "--distilled",
        "--prompt", request["prompt"],
        "--seed", str(int(request["seed"])),
        "--width", str(int(request["width"])),
        "--height", str(int(request["height"])),
        "--frames", str(int(request["frames"])),
        "--frame-rate", str(request.get("fps") or 24),
        "--output", output,
    ]
    # int8 holds about 26 GB of weights: on a Mac with 32 GB or less the transformer streams its blocks from the disk.
    if memory_gb() < 40:
        arguments.append("--low-ram")
    if request.get("audio") is False:
        arguments.append("--no-audio")
    for image in request.get("images") or []:
        path = os.path.join(files, image["file"])
        if not os.path.exists(path):
            print(f"Error: Bild {image['file'][:12]} fehlt auf dem Mac.", flush=True)
            raise SystemExit(1)
        # Files are named by hash only; with an extension neither PIL nor ffmpeg has to guess.
        named = os.path.join(work, f"{image.get('at') or 'first'}-{image['file'][:12]}.jpg")
        if not os.path.exists(named):
            os.symlink(path, named)
        strength = str(max(0.1, min(float(image.get("strength") or 1.0), 1.0)))
        if image.get("at") == "last":
            # "guide" holds the picture at the last frame and lets the frames before it move towards it.
            arguments += ["--image", named, "last", strength, "guide"]
        else:
            # At frame 0 the default binding replaces the first latent: the clip starts on exactly this picture.
            arguments += ["--image", named, "0", strength]
    return arguments


def with_ai_note(source: str, target: str, label: str, creator: str) -> str:
    """The clip with a note in its metadata that it is AI-made (EU AI Act Art. 50(2)), as paint.py's XMP does for
    pictures. The streams are copied, not encoded again; without ffmpeg the clip goes out as it is."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return source
    note = f"KI-generiert mit {label} ({creator}, img-o-mat)"
    done = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", source, "-map", "0", "-c", "copy",
         "-metadata", f"comment={note}", "-metadata", f"description={note}", "-metadata", "encoder=img-o-mat",
         "-movflags", "+faststart", target],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        print(done.stderr.strip(), flush=True)
        return source
    return target


def main() -> int:
    request = json.load(sys.stdin)
    values = common.settings()
    common.prepare_environment(values)
    # Only what is on the disk: a missing file is an error, never a download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TQDM_DISABLE"] = "0"
    os.environ["PATH"] = os.pathsep.join([*EXTRA_PATH, os.environ.get("PATH") or "/usr/bin:/bin"])
    if platform.system() == "Darwin" and not shutil.which("ffmpeg"):
        print("Error: ffmpeg fehlt auf dem Mac (brew install ffmpeg); ltx-2-mlx schreibt das Video damit.", flush=True)
        return 1
    emit({"event": "stage", "stage": "loading", "fraction": 0})
    work = tempfile.mkdtemp(prefix="img-o-mat-video-")
    try:
        raw = os.path.join(work, "clip.mp4")
        try:
            arguments = command(request, values, raw)
        except SystemExit as stop:
            return int(stop.code or 1)
        if not os.path.exists(arguments[0]):
            print("Error: ltx-2-mlx ist auf diesem Mac nicht installiert; install-worker.sh ausführen.", flush=True)
            return 1
        started = time.perf_counter()
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        bar, last = 0, -1.0
        pending = b""
        while True:
            chunk = process.stdout.read1(4096) if hasattr(process.stdout, "read1") else process.stdout.read(4096)
            if not chunk:
                break
            pending += chunk
            # tqdm redraws its line with \r; every piece is a line here.
            *lines, pending = re.split(rb"[\r\n]", pending)
            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                match = PROGRESS.search(line)
                if match and "%|" in line:
                    done, total = int(match[1]), int(match[2])
                    low, high = BARS[min(bar, len(BARS) - 1)]
                    fraction = round(low + (high - low) * done / max(total, 1), 3)
                    if fraction != last:
                        emit({"event": "stage", "stage": "painting", "fraction": fraction})
                        last = fraction
                    if done >= total:
                        bar += 1
                    continue
                print(line, flush=True)
        code = process.wait()
        finished = time.perf_counter()
        if code != 0 or not os.path.exists(raw):
            print(f"Error: ltx-2-mlx endete mit {code} ohne Video.", flush=True)
            return 1
        emit({"event": "stage", "stage": "saving", "fraction": 0.98})
        clip = with_ai_note(raw, os.path.join(work, "img-o-mat.mp4"), request.get("label") or request["model"],
                            request.get("creator") or "img-o-mat")
        with open(clip, "rb") as file:
            data = file.read()
        emit({
            "event": "done",
            "mp4": base64.b64encode(data).decode("ascii"),
            "loadSeconds": None,
            "paintSeconds": round(finished - started, 1),
            "peakMemoryGb": None,
        })
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
