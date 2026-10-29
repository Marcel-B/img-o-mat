"""The one command the service runs on the Mac that paints, over SSH (or as a child process where it paints itself):

    launch.py paint mflux     a picture, with that runner's Python
    launch.py fetch           a model's download, with mflux's Python
    launch.py probe           what this Mac has
    launch.py receive <sha>   a file on stdin into the files folder

Over SSH the service's key is only allowed this script (install-worker.sh writes it into ~/.ssh/authorized_keys as
restrict,command="/usr/bin/python3 <root>/worker/launch.py --ssh"): sshd runs it whatever the service asks and leaves
what was asked in SSH_ORIGINAL_COMMAND, of which only the words above count. So the key opens nothing else on
oi-o-mat, not even a shell.

For paint and fetch it reads the first line of stdin (the request), hands it to the script and then keeps reading its
own stdin: when that ends (the service cancelled and closed SSH, or the connection broke) it kills the script and its
children. Without that, a picture cut off over SSH would go on holding 20 GB until it next wrote a line. Plain
Python 3.9, macOS's own.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SCRIPTS = {"paint": "paint.py", "fetch": "fetch.py", "probe": "probe.py"}


def receive(digest: str) -> int:
    folder = common.settings()["files"]
    os.makedirs(folder, exist_ok=True)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        print("Error: kein SHA-256", file=sys.stderr)
        return 2
    hasher = hashlib.sha256()
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=".receive-")
    try:
        with os.fdopen(handle, "wb") as out:
            while chunk := sys.stdin.buffer.read(1 << 20):
                hasher.update(chunk)
                out.write(chunk)
        if hasher.hexdigest() != digest:
            print("Error: SHA-256 passt nicht", file=sys.stderr)
            return 1
        os.replace(temporary, os.path.join(folder, digest))
        return 0
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def run(script: str, python: str, folder: str) -> int:
    request = sys.stdin.buffer.readline()
    if not os.path.exists(python):
        print(f"Error: {python} fehlt; install.sh --worker auf diesem Mac ausführen.", flush=True)
        return 1
    child = subprocess.Popen(
        [python, "-u", os.path.join(folder, script)],
        stdin=subprocess.PIPE,
        # sshd passes no environment; the scripts find their root from it.
        env={**os.environ, "IMG_O_MAT_WORKER_ROOT": common.ROOT},
        start_new_session=True,
    )
    child.stdin.write(request)
    child.stdin.close()

    def watch() -> None:
        while sys.stdin.buffer.read(4096):
            pass
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    threading.Thread(target=watch, daemon=True).start()
    code = child.wait()
    # The watcher still blocks on stdin; a normal exit would wait for its lock and abort.
    sys.stdout.flush()
    os._exit(code)


def main() -> int:
    arguments = sys.argv[1:]
    if arguments[:1] == ["--ssh"]:
        # "img-o-mat paint mflux": the first word is only a name, the rest the command.
        try:
            arguments = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))[1:]
        except ValueError:
            arguments = []
    sys.argv = [sys.argv[0], *arguments]
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "receive" and len(sys.argv) > 2:
        return receive(sys.argv[2])
    if command == "probe":
        # Short and harmless; no watcher needed.
        return subprocess.call([sys.executable, os.path.join(common.HERE, SCRIPTS["probe"])])
    if command in ("paint", "fetch"):
        values = common.settings()
        runner = sys.argv[2] if len(sys.argv) > 2 else "mflux"
        python = values["flux3Python"] if runner == "flux3-mlx" else values["python"]
        return run(SCRIPTS[command], python, values["scripts"])
    print("Aufruf: launch.py paint <runner> | fetch | probe | receive <sha256>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
