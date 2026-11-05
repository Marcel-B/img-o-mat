"""The gallery as the service sees it: the pictures lie on the Mac that paints (worker/gallery.py, on the T7 there), the
service only passes them through over the same SSH key as the pictures. Only the page uses it; painting for Tonwerk-AI
and the gate does not touch it, and nothing is kept unless someone presses Speichern.

The small copies for the grid are kept here too (<root>/gallery-cache), since a name never changes its picture and a
grid of fifty would otherwise be fifty SSH connections each time the gallery opens.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

NAME = re.compile(r"\d{8}-\d{6}-[0-9a-f]{12}")
# The worker's exit codes (worker/gallery.py EXIT) as HTTP statuses; 255 is ssh's own.
STATUS = {2: 400, 3: 503, 4: 404, 255: 502}


class GalleryError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Gallery:
    def __init__(self, config, link, log):
        self.link = link
        self.log = log
        self.cache = os.path.join(config.root, "gallery-cache")

    def _run(self, *arguments: str, data: bytes = b"", timeout: float = 60) -> bytes:
        try:
            out = subprocess.run(self.link.command("gallery", *arguments), input=data, capture_output=True,
                                 timeout=timeout, env=self.link.environment())
        except (OSError, subprocess.SubprocessError) as error:
            raise GalleryError(502, f"{self.link.name} antwortet nicht: {error}")
        if out.returncode != 0:
            lines = out.stderr.decode(errors="replace").strip().splitlines()
            said = next((line[len("Error: "):] for line in reversed(lines) if line.startswith("Error: ")), None)
            if said is None:
                said = f"{self.link.name} antwortet nicht: {lines[-1] if lines else 'Exit ' + str(out.returncode)}"
            raise GalleryError(STATUS.get(out.returncode, 500), said)
        return out.stdout

    @staticmethod
    def checked(name: str) -> str:
        if not NAME.fullmatch(name):
            raise GalleryError(404, "Dieses Bild gibt es in der Galerie nicht.")
        return name

    def list(self) -> dict:
        return json.loads(self._run("list"))

    def save(self, jpeg_b64: str, meta: dict) -> dict:
        saved = json.loads(self._run("save", data=json.dumps({"jpeg": jpeg_b64, "meta": meta}).encode()))
        if saved.get("created"):
            self.log(f"Bild {saved['name']} in die Galerie gelegt ({meta.get('app')}, {meta.get('model')}).")
        return saved

    def picture(self, name: str, thumb: bool) -> bytes:
        name = self.checked(name)
        cached = os.path.join(self.cache, name + ".jpg")
        if thumb:
            try:
                with open(cached, "rb") as file:
                    return file.read()
            except FileNotFoundError:
                pass
        data = self._run("get", name, *(["thumb"] if thumb else []))
        if thumb:
            os.makedirs(self.cache, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=self.cache, prefix=".get-")
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            os.replace(temporary, cached)
        return data

    def delete(self, name: str) -> dict:
        name = self.checked(name)
        deleted = json.loads(self._run("delete", name))
        try:
            os.remove(os.path.join(self.cache, name + ".jpg"))
        except FileNotFoundError:
            pass
        self.log(f"Bild {name} aus der Galerie gelöscht.")
        return deleted
