"""LoRAs and reference pictures a request uses, kept by their SHA-256: a client uploads a file once
(PUT /v1/files/<sha256>, HEAD tells whether it is there) and names it in every request after. A LoRA of a few hundred
MB then crosses the network once, and no request carries a path on this Mac. Files nobody used for keepFilesDays go."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time

HASH = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20
# A LoRA for the larger models stays well under this; anything bigger is a mistake.
MAX_BYTES = 4 << 30


class FileError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Files:
    def __init__(self, config, log):
        self.folder = config.files
        self.keep = config.keep_files
        self.log = log
        os.makedirs(self.folder, exist_ok=True)

    def path(self, digest: str) -> str | None:
        if not HASH.match(digest or ""):
            return None
        path = os.path.join(self.folder, digest)
        return path if os.path.exists(path) else None

    def use(self, digest: str) -> str:
        """The file's path for a picture; touched, so it is kept while it is used."""
        path = self.path(digest)
        if path is None:
            raise FileError(400, f"Datei {digest} ist nicht da; erst mit PUT /v1/files/<sha256> hochladen.")
        os.utime(path)
        return path

    def store(self, digest: str, stream, length: int) -> bool:
        """Writes the upload and checks its hash; False when it was there already."""
        if not HASH.match(digest or ""):
            raise FileError(400, "Der Name ist der SHA-256 der Datei in Kleinbuchstaben.")
        if length > MAX_BYTES:
            raise FileError(413, "Die Datei ist zu groß.")
        if self.path(digest):
            # The body still has to be read, or the connection is left with it.
            while length > 0:
                length -= len(stream.read(min(CHUNK, length)) or b"") or length
            os.utime(os.path.join(self.folder, digest))
            return False
        hasher = hashlib.sha256()
        handle, temporary = tempfile.mkstemp(dir=self.folder, prefix=".upload-")
        try:
            with os.fdopen(handle, "wb") as out:
                left = length
                while left > 0:
                    chunk = stream.read(min(CHUNK, left))
                    if not chunk:
                        raise FileError(400, "Die Datei kam nicht ganz an.")
                    hasher.update(chunk)
                    out.write(chunk)
                    left -= len(chunk)
            if hasher.hexdigest() != digest:
                raise FileError(400, "Der SHA-256 passt nicht zum Inhalt.")
            os.replace(temporary, os.path.join(self.folder, digest))
            return True
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def prune(self) -> None:
        limit = time.time() - self.keep
        for name in os.listdir(self.folder):
            path = os.path.join(self.folder, name)
            try:
                if os.path.getmtime(path) < limit:
                    os.remove(path)
                    self.log(f"Datei {name} lange nicht benutzt, gelöscht.")
            except OSError:
                pass
