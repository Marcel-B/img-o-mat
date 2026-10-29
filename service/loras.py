"""The LoRAs for the image models, moved from Tonwerk-AI's ImageLoras.cs: .safetensors files in <root>/loras/
(uploaded, fetched from Hugging Face, or put there by hand) and what is known about each in loras.json beside them
(label, the model it fits, trigger words, strength, where it came from, its SHA-256). A file without an entry gets what
its header says: trainers like ai-toolkit and kohya note the base model, and FLUX.2's layer widths give away its size.

A picture names a LoRA by its name; the service copies the file to the Mac that paints by its hash, once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

EXTENSION = ".safetensors"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,99}$")
MAX_BYTES = 2 << 30
MAX_PER_PICTURE = 3
MAX_STRENGTH = 2.0


def krea(name: str, label: str, trigger: str) -> dict:
    return {
        "name": f"krea-{name}", "label": label, "model": "krea-2-turbo", "trigger": trigger, "strength": 1,
        "repo": f"krea/Krea-2-LoRA-{name}", "file": f"{name}.safetensors", "license": "Krea 2 Community",
    }


# Krea's own style LoRAs for Krea 2 (huggingface.co/collections/krea/krea-2-loras), with the trigger phrases and the
# strength their model cards give. Klein, Z-Image and FLUX 3 have no official ones.
SUGGESTIONS = [
    krea("retroanime", "Retro-Anime", "Purple retro anime style"),
    krea("sunsetblur", "Sonnenuntergang verwischt", "ethereal motion blur style"),
    krea("vintagetarot", "Vintage-Tarot", "vintage tarot style"),
    krea("rainywindow", "Regenfenster", "rainy window style"),
    krea("darkbrush", "Tuschpinsel", "monochrome ink wash style"),
    krea("dotmatrix", "Punktraster", "Monochrome stippling style"),
    krea("neondrip", "Neon Drip", "Textured abstract style"),
    krea("kidsdrawing", "Kinderzeichnung", "naive expressive sketch style"),
    krea("softwatercolor", "Art-déco-Aquarell", "Art Deco watercolor style"),
]


class LoraError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def valid_name(name: str) -> bool:
    return bool(NAME.match(name or "")) and ".." not in name


def read_header(path: str) -> dict | None:
    """A safetensors file's header (tensors' names and shapes plus __metadata__), or None when it is not one: eight
    bytes of header length (little endian), then the header as a JSON object."""
    try:
        with open(path, "rb") as file:
            prefix = file.read(8)
            if len(prefix) < 8:
                return None
            (length,) = struct.unpack("<Q", prefix)
            if length < 2 or length > 100_000_000 or length > os.path.getsize(path) - 8:
                return None
            header = json.loads(file.read(length))
            return header if isinstance(header, dict) else None
    except (OSError, ValueError):
        return None


def family(note: str) -> str | None:
    text = "".join(c for c in note.lower() if c.isalnum())
    if "klein4b" in text or "f2k4b" in text or "flux2klein4b" in text:
        return "klein-4b"
    if "klein9b" in text or "f2k9b" in text:
        return "klein-9b"
    if "zimage" in text:
        return "z-image-turbo"
    if "krea" in text:
        return "krea-2-turbo"
    if "flux3" in text:
        return "flux3-klein"
    return None


def detect_model(path: str, header: dict, model_ids: set[str]) -> str | None:
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    notes = [metadata[k] for k in ("ss_base_model_version", "modelspec.architecture", "base_model", "ss_sd_model_name")
             if isinstance(metadata.get(k), str)]
    notes.append(os.path.splitext(os.path.basename(path))[0])
    found = next((f for f in map(family, notes) if f), None)
    if found is None:
        for name, tensor in header.items():
            # FLUX.2's single blocks fuse attention and MLP into to_qkv_mlp_proj; its input width is the model's.
            if ("to_qkv_mlp_proj" in name and name.endswith(("lora_down.weight", "lora_A.weight"))
                    and isinstance(tensor, dict) and len(tensor.get("shape") or []) == 2):
                found = {3072: "klein-4b", 4096: "klein-9b"}.get(tensor["shape"][1])
                break
    return found if found in model_ids else None


def trigger_of(header: dict) -> str | None:
    """A trigger word some trainers note (Kohya's ss_trigger, diffusers' instance prompt)."""
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    for key in ("trigger_word", "ss_trigger", "instance_prompt", "trigger"):
        if isinstance(metadata.get(key), str) and metadata[key].strip():
            return metadata[key].strip()
    return None


def sha256_of(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


class Loras:
    def __init__(self, config, catalogue, log):
        self.config = config
        self.catalogue = catalogue
        self.log = log
        self.folder = os.path.join(config.root, "loras")
        self.meta_file = os.path.join(self.folder, "loras.json")
        self.fetching: set[str] = set()
        self._lock = threading.RLock()
        os.makedirs(self.folder, exist_ok=True)

    def _rows(self) -> dict:
        try:
            with open(self.meta_file) as file:
                return json.load(file)
        except (FileNotFoundError, ValueError):
            return {}

    def _store(self, rows: dict) -> None:
        temporary = f"{self.meta_file}.{os.getpid()}.tmp"
        with open(temporary, "w") as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)
        os.replace(temporary, self.meta_file)

    def path_for(self, name: str) -> str | None:
        if not valid_name(name):
            return None
        path = os.path.join(self.folder, name + EXTENSION)
        return path if os.path.exists(path) else None

    def describe(self, name: str, row: dict | None = None) -> dict | None:
        path = self.path_for(name)
        if path is None:
            return None
        row = row if row is not None else self._rows().get(name)
        stat = os.stat(path)
        if row is None or row.get("size") != stat.st_size or row.get("mtime") != int(stat.st_mtime):
            # Put there by hand, or replaced: read the header again, keep what was set by hand.
            header = read_header(path)
            if header is None:
                return None  # half copied, or not a safetensors file
            detected = {"model": detect_model(path, header, {m["id"] for m in self.catalogue.models}),
                        "trigger": trigger_of(header)}
            row = {**detected, **{k: v for k, v in (row or {}).items() if v is not None},
                   "size": stat.st_size, "mtime": int(stat.st_mtime), "sha256": sha256_of(path)}
            with self._lock:
                rows = self._rows()
                rows[name] = row
                self._store(rows)
        return {
            "name": name,
            "label": row.get("label") or name,
            "model": row.get("model"),
            "trigger": row.get("trigger"),
            "strength": row.get("strength") if row.get("strength") is not None else 1,
            "source": row.get("source"),
            "sizeBytes": stat.st_size,
            "sha256": row["sha256"],
            "modifiedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        }

    def list(self) -> dict:
        rows = self._rows()
        names = sorted(f[: -len(EXTENSION)] for f in os.listdir(self.folder)
                       if f.endswith(EXTENSION) and valid_name(f[: -len(EXTENSION)]))
        loras = [d for d in (self.describe(n, rows.get(n)) for n in names) if d]
        loras.sort(key=lambda l: l["label"].lower())
        have = {l["name"].lower() for l in loras}
        with self._lock:
            fetching = sorted(self.fetching)
        return {
            "loras": loras,
            "suggestions": [s for s in SUGGESTIONS if s["name"].lower() not in have],
            "fetching": fetching,
        }

    def save(self, name: str, stream, length: int, settings: dict, source: str | None = None) -> dict:
        if not valid_name(name):
            raise LoraError(400, "Ungültiger Name: Buchstaben, Ziffern, Leerzeichen, . _ -, bis 100 Zeichen.")
        if length > MAX_BYTES:
            raise LoraError(413, "Die Datei ist zu groß.")
        path = os.path.join(self.folder, name + EXTENSION)
        # A picture might read the old file while the new one arrives: it only ever sees a complete file.
        handle, temporary = tempfile.mkstemp(dir=self.folder, prefix=".part-")
        try:
            with os.fdopen(handle, "wb") as out:
                left = length
                while left is None or left > 0:
                    chunk = stream.read(1 << 20 if left is None else min(1 << 20, left))
                    if not chunk:
                        if left is None:
                            break
                        raise LoraError(400, "Die Datei kam nicht ganz an.")
                    out.write(chunk)
                    if left is not None:
                        left -= len(chunk)
            if read_header(temporary) is None:
                raise LoraError(400, "Das ist keine safetensors-Datei.")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        with self._lock:
            rows = self._rows()
            rows[name] = {"source": source, **{k: v for k, v in self._clean(settings).items() if v is not None}}
            self._store(rows)
        self.log(f"LoRA {name} gespeichert.")
        return self.describe(name)

    def update(self, name: str, settings: dict) -> dict | None:
        if self.describe(name) is None:
            return None
        with self._lock:
            rows = self._rows()
            row = rows.get(name, {})
            for key, value in self._clean(settings).items():
                row[key] = value
            rows[name] = row
            self._store(rows)
        return self.describe(name)

    def delete(self, name: str) -> bool:
        path = self.path_for(name)
        if path is None:
            return False
        os.remove(path)
        with self._lock:
            rows = self._rows()
            rows.pop(name, None)
            self._store(rows)
        return True

    def fetch(self, suggestion: dict) -> dict | None:
        """Downloads a suggestion from Hugging Face; None when it is being fetched already. Runs to the end even when
        the request that asked gives up."""
        with self._lock:
            if suggestion["name"] in self.fetching:
                return None
            self.fetching.add(suggestion["name"])
        try:
            url = f"https://huggingface.co/{suggestion['repo']}/resolve/main/{urllib.parse.quote(suggestion['file'])}"
            request = urllib.request.Request(url)
            if token := self.config.token():
                request.add_header("Authorization", f"Bearer {token}")
            try:
                with urllib.request.urlopen(request, timeout=60) as answer:
                    length = int(answer.headers.get("content-length") or 0) or None
                    settings = {k: suggestion.get(k) for k in ("label", "model", "trigger", "strength")}
                    return self.save(suggestion["name"], answer, length, settings, suggestion["repo"])
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise LoraError(502, f"Hugging Face verweigert {suggestion['repo']}: Lizenz dort annehmen und einen Token hinterlegen.")
                raise LoraError(502, f"Hugging Face antwortet {error.code} für {suggestion['repo']}.")
            except urllib.error.URLError as error:
                raise LoraError(502, f"Hugging Face nicht erreichbar: {error.reason}")
        finally:
            with self._lock:
                self.fetching.discard(suggestion["name"])

    def _clean(self, settings: dict) -> dict:
        """What may be set by hand; an empty text clears a field, a missing one leaves it."""
        clean = {}
        for key in ("label", "model", "trigger"):
            if key in settings:
                value = settings[key]
                clean[key] = value.strip() if isinstance(value, str) and value.strip() else None
        if clean.get("model") and not self.catalogue.find(clean["model"]):
            raise LoraError(400, f"Unbekanntes Modell {clean['model']}.")
        if "strength" in settings and settings["strength"] is not None:
            clean["strength"] = max(0.0, min(float(settings["strength"]), MAX_STRENGTH))
        return clean
