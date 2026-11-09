"""Makes one video clip for img-o-mat with ltx-2-mlx (LTX-2.3) or minimax-h3-mlx. Run with the runner environment's Python
(install-worker.sh); the service starts it like paint.py, one clip per process, over SSH on the Mac that paints.

Both runners use their own command line; ltx-2-mlx runs (`ltx-2-mlx generate`), from the local snapshots only: the models came onto the
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
from collections import deque
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


def snapshot(repo: str, allow_patterns: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        # Filtered downloads intentionally omit other components in the same repository.
        return snapshot_download(repo, local_files_only=True, allow_patterns=allow_patterns)
    except Exception as error:
        print(f"Error: {repo} ist lokal nicht verfügbar; auf der Seite unter Modelle → Videos herunterladen."
              f" Ursache: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(1)


def memory_gb() -> float:
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / (1 << 30)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def runner_error(lines) -> str | None:
    lines = list(lines)
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if "CalledProcessError:" in line:
            continue  # This only wraps the earlier failure of a low-memory phase.
        if re.match(r"^(?:[\w.]*Error|[\w.]*Exception):", line):
            if line.endswith(":") and index + 1 < len(lines):
                return line + " " + lines[index + 1]
            return line
    return None


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


def minimax_command(request: dict, values: dict, output: str) -> list[str]:
    def component(repo: str) -> str:
        return snapshot(repo, (request.get("allowPatterns") or {}).get(repo))

    arguments = [
        sys.executable, "-u", os.path.join(common.HERE, "minimax_generate.py"),
        "--source", values["minimaxSource"],
        "--quantized-text-encoder", component(request["textEncoder"]),
        "--output", output,
        "--checkpoint", os.path.join(component(request["checkpoint"]), "FL2VA"),
        "--transformer", component(request["model"]),
        "--duration", str(request["duration"]),
        # The upstream CLI counts sigma grid points: nine give eight denoising forwards.
        "--steps", "17" if request.get("quality") == "high" else "9",
        "--seed", str(int(request["seed"])),
        "--width", str(int(request["width"])), "--height", str(int(request["height"])),
    ]
    if request.get("lowRam"):
        arguments += ["--low-ram", "--streaming-source", values["minimaxStreamingSource"]]
    for image in request.get("images") or []:
        path = os.path.join(values["files"], image["file"])
        if not os.path.isfile(path):
            print(f"Error: Bild {image['file'][:12]} fehlt auf dem Mac.", flush=True)
            raise SystemExit(1)
        arguments += ["--image", path, "--anchor", image.get("at") or "first"]
    # A prompt beginning with a dash is still text, never a CLI option.
    return [*arguments, "--", request["prompt"]]


def progress(line: str, minimax: bool, bar: int) -> tuple[float, bool] | None:
    match = re.search(r"step (\d+)/(\d+)", line) if minimax else PROGRESS.search(line)
    if not match or (not minimax and "%|" not in line):
        return None
    done, total = int(match[1]), int(match[2])
    low, high = (0.05, 0.95) if minimax else BARS[min(bar, len(BARS) - 1)]
    return round(low + (high - low) * done / max(total, 1), 3), done >= total


def with_ai_note(source: str, target: str, label: str, creator: str, audio: bool = True) -> str:
    """The clip with a note in its metadata that it is AI-made (EU AI Act Art. 50(2)), as paint.py's XMP does for
    pictures. The streams are copied, not encoded again; without ffmpeg the clip goes out as it is."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if not audio:
            raise RuntimeError("ffmpeg fehlt; der Ton kann nicht entfernt werden.")
        return source
    note = f"KI-generiert mit {label} ({creator}, img-o-mat)"
    done = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", source, "-map", "0" if audio else "0:v:0", "-c", "copy",
         "-metadata", f"comment={note}", "-metadata", f"description={note}", "-metadata", "encoder=img-o-mat",
         "-movflags", "+faststart", target],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        if not audio:
            raise RuntimeError("ffmpeg konnte den Ton nicht entfernen: " + done.stderr.strip())
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
        print("Error: ffmpeg fehlt auf dem Mac (brew install ffmpeg); die Videorunner schreiben das Video damit.", flush=True)
        return 1
    minimax = request.get("runner") == "minimax-h3-mlx"
    runner = "minimax-h3-mlx" if minimax else "ltx-2-mlx"
    emit({"event": "stage", "stage": "loading", "fraction": 0})
    work = os.environ.get("IMG_O_MAT_VIDEO_WORK") or tempfile.mkdtemp(prefix="img-o-mat-video-")
    try:
        raw = os.path.join(work, "clip.mp4")
        try:
            arguments = (minimax_command if minimax else command)(request, values, raw)
        except SystemExit as stop:
            return int(stop.code or 1)
        if not os.path.exists(arguments[2] if minimax else arguments[0]):
            print(f"Error: {runner} ist auf diesem Mac nicht installiert; install-worker.sh ausführen.", flush=True)
            return 1
        started = time.perf_counter()
        os.environ["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        bar, last = 0, -1.0
        peak_memory = None
        diagnostics = deque(maxlen=80)
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
                if minimax and line.startswith("MINIMAX_PEAK_GB="):
                    peak_memory = max(peak_memory or 0, float(line.split("=", 1)[1]))
                    continue
                value = progress(line, minimax, bar)
                if value:
                    fraction, complete = value
                    if fraction != last:
                        emit({"event": "stage", "stage": "painting", "fraction": fraction})
                        last = fraction
                    if complete:
                        bar += 1
                    continue
                print(line, flush=True)
                diagnostics.append(line)
        if pending.strip():
            line = pending.decode("utf-8", errors="replace").strip()
            print(line, flush=True)
            diagnostics.append(line)
        code = process.wait()
        process.stdout.close()
        finished = time.perf_counter()
        if code != 0 or not os.path.exists(raw):
            cause = runner_error(diagnostics)
            print(f"Error: {runner}: {cause}" if cause else f"Error: {runner} endete mit {code} ohne Video.", flush=True)
            return 1
        emit({"event": "stage", "stage": "saving", "fraction": 0.98})
        try:
            clip = with_ai_note(raw, os.path.join(work, "img-o-mat.mp4"), request.get("label") or request["model"],
                                request.get("creator") or "img-o-mat", audio=request.get("audio") is not False)
        except RuntimeError as error:
            print(f"Error: {error}", flush=True)
            return 1
        with open(clip, "rb") as file:
            data = file.read()
        emit({
            "event": "done",
            "mp4": base64.b64encode(data).decode("ascii"),
            "loadSeconds": None,
            "paintSeconds": round(finished - started, 1),
            "peakMemoryGb": peak_memory,
        })
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
