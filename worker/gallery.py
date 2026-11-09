"""The gallery: pictures someone chose to keep, on the Mac that paints, on the same disk as the models (the T7 on
oi-o-mat), so they survive the service's container and need no space there. launch.py runs it for the service:

    launch.py gallery list                 {"pictures": [...], "folder": ..., "disk": "T7"} on stdout
    launch.py gallery save                 {"jpeg": "<base64>", "meta": {...}} on stdin, {"name", "created"} out
    launch.py gallery get <name> [thumb]   the JPEG on stdout
    launch.py gallery ref <sha256>         a reference picture the gallery keeps, on stdout
    launch.py gallery delete <name>

A picture is <name>.jpg with <name>.json beside it (model, seed, prompt, LoRAs, references, ...) and a small copy in
.thumbs/ for the page's grid. The pictures it was painted from lie in .refs/ by their SHA-256, once for every picture
that used them, and go with the last picture that names them. The name is the time it was saved and the start of its SHA-256, so the same picture is kept only once.
Errors are an "Error: ..." line on stderr; the exit code tells the service which kind (see EXIT). Plain Python 3.9,
macOS's own, like probe.py: nothing here needs the runners.
"""

from __future__ import annotations

import base64
import binascii
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

NAME = re.compile(r"\d{8}-\d{6}-[0-9a-f]{12}")
THUMB_PIXELS = 512
# What the service may store beside a picture; anything else in the request is dropped.
META = {"model": 60, "prompt": 2000, "negativePrompt": 2000, "paintedPrompt": 4000, "size": 16, "app": 64, "creator": 60}
MAX_LORAS = 3
MAX_REFERENCES = 4
ROLES = ("style", "figure", "place", "start")
HASH = re.compile(r"[0-9a-f]{64}")
EXIT = {"usage": 2, "disk": 3, "missing": 4}


class Refused(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def folder() -> str:
    """The gallery's folder, after checking that its disk is there: an unplugged T7 leaves /Volumes/T7 missing or an
    empty folder on the Mac's own disk, and writing there would fill the system disk with pictures nobody finds."""
    values = common.settings()
    path = values["gallery"]
    disk = values["galleryDisk"]
    if disk and not os.path.ismount(os.path.join("/Volumes", disk)):
        raise Refused("disk", f"Die Platte {disk} ist nicht angeschlossen.")
    os.makedirs(os.path.join(path, ".thumbs"), exist_ok=True)
    os.makedirs(os.path.join(path, ".refs"), exist_ok=True)
    return path


def checked(name: str) -> str:
    if not NAME.fullmatch(name or ""):
        raise Refused("usage", "Kein Bildname.")
    return name


def read_meta(path: str, name: str) -> dict:
    try:
        with open(os.path.join(path, name + ".json")) as file:
            meta = json.load(file)
    except (OSError, ValueError):
        meta = {}
    try:
        size = os.path.getsize(os.path.join(path, name + ".jpg"))
    except OSError:
        size = None
    return {**meta, "name": name, "bytes": size}


def listing() -> dict:
    path = folder()
    names = sorted((os.path.basename(p)[:-4] for p in glob.glob(os.path.join(path, "*.jpg"))), reverse=True)
    values = common.settings()
    return {"pictures": [read_meta(path, n) for n in names if NAME.fullmatch(n)], "folder": path, "disk": values["galleryDisk"]}


def thumbnail(source: str, target: str) -> None:
    """A small copy for the grid with macOS's sips; without it (or when it fails) the page gets the picture itself."""
    if not os.path.exists("/usr/bin/sips"):
        return
    try:
        subprocess.run(["/usr/bin/sips", "-Z", str(THUMB_PIXELS), "-s", "format", "jpeg", "-s", "formatOptions", "80",
                        source, "--out", target], capture_output=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        try:
            os.remove(target)
        except OSError:
            pass


def write(path: str, data: bytes) -> None:
    """Whole or not at all: a picture cut off by a full disk or a pulled cable must not be listed."""
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".save-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def number(value, low: float, high: float):
    try:
        return max(low, min(float(value), high))
    except (TypeError, ValueError):
        return None


def loras_of(given) -> list:
    """The LoRAs as the page names them: file name in the library, label, strength."""
    out = []
    for lora in (given if isinstance(given, list) else [])[:MAX_LORAS]:
        if isinstance(lora, dict) and lora.get("name"):
            out.append({"name": str(lora["name"])[:120], "label": str(lora.get("label") or lora["name"])[:120],
                        "strength": number(lora.get("strength"), 0, 4)})
    return out


def keep_references(path: str, given) -> list:
    """Writes the pictures a painting looked at or painted over into .refs/ and says which they were. One that did not
    come along (the service no longer had it) is still named, so the page can say a picture was used."""
    out = []
    for reference in (given if isinstance(given, list) else [])[:MAX_REFERENCES]:
        if not isinstance(reference, dict):
            continue
        entry = {"role": reference.get("role") if reference.get("role") in ROLES else "style",
                 "strength": number(reference.get("strength"), 0.1, 0.9)}
        try:
            jpeg = base64.b64decode(reference.get("jpeg") or "", validate=True)
        except (binascii.Error, ValueError):
            jpeg = b""
        if jpeg.startswith(b"\xff\xd8\xff"):
            digest = hashlib.sha256(jpeg).hexdigest()
            target = os.path.join(path, ".refs", digest + ".jpg")
            if not os.path.exists(target):
                write(target, jpeg)
            entry["sha256"] = digest
        out.append(entry)
    return out


def save(asked: dict) -> dict:
    try:
        jpeg = base64.b64decode(asked.get("jpeg") or "", validate=True)
    except (binascii.Error, ValueError):
        raise Refused("usage", "Das Bild ist kein Base64.")
    if not jpeg.startswith(b"\xff\xd8\xff"):
        raise Refused("usage", "Das ist kein JPEG.")
    path = folder()
    digest = hashlib.sha256(jpeg).hexdigest()
    for existing in glob.glob(os.path.join(path, f"*-{digest[:12]}.jpg")):
        return {"name": os.path.basename(existing)[:-4], "created": False}
    name = time.strftime("%Y%m%d-%H%M%S") + "-" + digest[:12]
    given = asked.get("meta") or {}
    meta = {key: str(given[key])[:limit] for key, limit in META.items() if given.get(key) not in (None, "")}
    if isinstance(given.get("seed"), int):
        meta["seed"] = given["seed"]
    # How long it took, as the worker measured it: loading the model and painting.
    for key in ("loadSeconds", "paintSeconds"):
        if number(given.get(key), 0, 7 * 86400) is not None:
            meta[key] = round(number(given.get(key), 0, 7 * 86400), 1)
    if loras_of(given.get("loras")):
        meta["loras"] = loras_of(given.get("loras"))
    references = keep_references(path, asked.get("references"))
    if references:
        meta["references"] = references
    meta.update({"sha256": digest, "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    picture = os.path.join(path, name + ".jpg")
    write(picture, jpeg)
    write(os.path.join(path, name + ".json"), json.dumps(meta, ensure_ascii=False, indent=2).encode())
    thumbnail(picture, os.path.join(path, ".thumbs", name + ".jpg"))
    return {"name": name, "created": True}


def get(name: str, thumb: bool) -> bytes:
    path = folder()
    name = checked(name)
    candidates = [os.path.join(path, ".thumbs", name + ".jpg")] if thumb else []
    for candidate in candidates + [os.path.join(path, name + ".jpg")]:
        try:
            with open(candidate, "rb") as file:
                return file.read()
        except FileNotFoundError:
            continue
    raise Refused("missing", "Dieses Bild gibt es in der Galerie nicht.")


def reference(digest: str) -> bytes:
    path = folder()
    if not HASH.fullmatch(digest or ""):
        raise Refused("usage", "Kein SHA-256.")
    try:
        with open(os.path.join(path, ".refs", digest + ".jpg"), "rb") as file:
            return file.read()
    except FileNotFoundError:
        raise Refused("missing", "Dieses Vorlagenbild gibt es in der Galerie nicht.")


def named_references(path: str) -> set | None:
    """The references the gallery's pictures still name; None when one could not be read, so nothing is pruned."""
    names = set()
    for meta_path in glob.glob(os.path.join(path, "*.json")):
        try:
            with open(meta_path) as file:
                for entry in json.load(file).get("references") or []:
                    if isinstance(entry, dict) and entry.get("sha256"):
                        names.add(entry["sha256"])
        except (OSError, ValueError, AttributeError):
            return None
    return names


def delete(name: str) -> dict:
    path = folder()
    name = checked(name)
    picture = os.path.join(path, name + ".jpg")
    if not os.path.exists(picture):
        raise Refused("missing", "Dieses Bild gibt es in der Galerie nicht.")
    # The picture first: without it the rest is never listed, whatever else fails.
    for leftover in (picture, os.path.join(path, name + ".json"), os.path.join(path, ".thumbs", name + ".jpg")):
        try:
            os.remove(leftover)
        except FileNotFoundError:
            pass
    still = named_references(path)
    if still is not None:
        for leftover in glob.glob(os.path.join(path, ".refs", "*.jpg")):
            if os.path.basename(leftover)[:-4] not in still:
                try:
                    os.remove(leftover)
                except OSError:
                    pass
    return {"deleted": name}


def main(arguments: list) -> int:
    command = arguments[0] if arguments else ""
    try:
        if command == "list":
            json.dump(listing(), sys.stdout, ensure_ascii=False)
        elif command == "save":
            json.dump(save(json.loads(sys.stdin.buffer.read() or b"{}")), sys.stdout)
        elif command == "get" and len(arguments) > 1:
            sys.stdout.buffer.write(get(arguments[1], arguments[2:3] == ["thumb"]))
        elif command == "ref" and len(arguments) > 1:
            sys.stdout.buffer.write(reference(arguments[1]))
        elif command == "delete" and len(arguments) > 1:
            json.dump(delete(arguments[1]), sys.stdout)
        else:
            raise Refused("usage", "Aufruf: launch.py gallery list | save | get <name> [thumb] | ref <sha256> | delete <name>")
    except Refused as refusal:
        print(f"Error: {refusal.message}", file=sys.stderr)
        return EXIT[refusal.kind]
    except ValueError:
        print("Error: Die Anfrage ist kein JSON.", file=sys.stderr)
        return EXIT["usage"]
    except OSError as error:
        print(f"Error: {error.strerror or error} ({common.settings()['gallery']})", file=sys.stderr)
        return 1
    sys.stdout.flush()
    return 0
