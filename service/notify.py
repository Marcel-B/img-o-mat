"""A message to Onkel Bummel (Marcel's notification service, Marcel-B/onkel-bummel) when a picture is done or failed,
with the pictures attached, so nobody has to keep the page open while a model paints.

    "onkelBummel": {"url": "https://onkel-bummel.example", "tokenFile": "/etc/img-o-mat/onkel-bummel.token",
                    "topic": "img-o-mat", "click": "http://img-o-mat:1236/", "apps": null}

Without "url" nothing is sent. The token (a publish token `ob_…` for the topic) is read from "tokenFile" for every
message, so it can be put there later without a restart; "token" in the config works too. "apps" limits the messages
to these apps (null: every app); a request can also say "notify": false. Cancelled pictures send nothing.

Sending runs in its own thread and never holds up the queue: a message that does not get through after a few tries
is only logged. The job's id is the Idempotency-Key, so a retry after a lost answer makes no second message.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request

ATTEMPTS = 3
TIMEOUT = 30
PROMPT_CHARS = 300


class Notifier:
    def __init__(self, settings: dict | None, log, opener=urllib.request.urlopen):
        self.settings = settings or {}
        self.log = log
        self.opener = opener

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("url"))

    def token(self) -> str | None:
        if path := self.settings.get("tokenFile"):
            try:
                with open(path) as file:
                    if token := file.read().strip():
                        return token
            except OSError:
                pass
        return self.settings.get("token") or None

    def wanted(self, job) -> bool:
        if not self.enabled or job.cancelled or not job.request.get("notify", True):
            return False
        apps = self.settings.get("apps")
        return apps is None or job.app in apps

    def job_finished(self, job) -> None:
        """Called by the painter once a job is done or failed; returns at once."""
        if not self.wanted(job):
            return
        if job.error and job.error[0] == 499:
            return
        message = self.message_of(job)
        threading.Thread(target=self.send, args=(message, job.id), daemon=True, name="onkel-bummel").start()

    def message_of(self, job) -> dict:
        model = job.model["label"]
        prompt = job.request["prompt"]
        if len(prompt) > PROMPT_CHARS:
            prompt = prompt[: PROMPT_CHARS - 1].rstrip() + "…"
        message = {"topic": self.settings.get("topic") or "img-o-mat"}
        if job.error:
            message.update(title=f"Bild fehlgeschlagen ({model})", priority=4, tags=["warning", "img-o-mat"],
                           message=f"{job.error[1]}\n\n{job.app}: {prompt}")
        else:
            count = len(job.results)
            seconds = sum((r.get("loadSeconds") or 0) + (r.get("paintSeconds") or 0) for r in job.results)
            took = f" in {seconds / 60:.0f} min" if seconds >= 90 else f" in {seconds:.0f} s" if seconds else ""
            message.update(title=f"{count} Bilder fertig ({model})" if count > 1 else f"Bild fertig ({model})",
                           priority=3, tags=["white_check_mark", "img-o-mat"],
                           message=f"{job.app}{took}: {prompt}",
                           images=[{"data": base64.b64encode(r["jpeg"]).decode("ascii"),
                                    "filename": f"{job.id}-{index + 1}.jpg"} for index, r in enumerate(job.results)])
        if click := self.settings.get("click"):
            message["click"] = click
        return message

    def send(self, message: dict, key: str) -> bool:
        token = self.token()
        if not token:
            self.log("Onkel Bummel: kein Token (onkelBummel.tokenFile oder token), Nachricht nicht gesendet.")
            return False
        url = self.settings["url"].rstrip("/") + "/api/v1/messages"
        body = json.dumps(message).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}", "Idempotency-Key": key}
        for attempt in range(1, ATTEMPTS + 1):
            try:
                with self.opener(urllib.request.Request(url, data=body, headers=headers, method="POST"), timeout=TIMEOUT):
                    return True
            except urllib.error.HTTPError as error:
                # A wrong token, topic or picture stays wrong; only the server's own trouble is worth another try.
                if error.code < 500:
                    self.log(f"Onkel Bummel: {error.code} {error.reason}, Nachricht nicht gesendet.")
                    return False
                reason = f"{error.code} {error.reason}"
            except (OSError, ValueError) as error:
                reason = str(error)
            if attempt < ATTEMPTS:
                time.sleep(5 * attempt)
        self.log(f"Onkel Bummel nicht erreichbar ({reason}), Nachricht nicht gesendet.")
        return False
