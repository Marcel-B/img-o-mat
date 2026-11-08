# img-o-mat

Die Bildmodelle von oi-o-mat als Dienst für alle Apps: FLUX.2 Klein 4B und 9B, Z-Image Turbo, Krea 2 Turbo und FLUX 3 Klein, mit Bild-LoRAs, Referenzbildern und einer Seite, auf der man Modelle und LoRAs verwaltet und direkt Prompts malt. Bisher hat Tonwerk-AI diese Modelle selbst gestartet; jetzt kann jede App sie nutzen, auch Tonwerk.

## Aufbau

```
 Apps ──► LLM-Schranke (oi-o-mat:1235) ──┐
                                         ├──► img-o-mat (Proxmox-Container, :1236) ──ssh──► oi-o-mat malt
 Tonwerk-AI (direkt, mit Vorrang) ───────┘        Seite, Warteschlange, LoRAs              Modelle auf der T7
                                                                                         Galerie auf der T7
```

- **Der Dienst** läuft in einem kleinen Debian-Container auf Proxmox (512 MB). Dort liegen die Seite, die Warteschlange, die LoRA-Bibliothek und hochgeladene Referenzbilder. Er ist reines Python ohne Pakete.
- **Gemalt wird auf oi-o-mat.** Für jedes Bild startet der Dienst per SSH einen Prozess mit mflux oder flux3-mlx, der das Modell lädt, malt und wieder geht. So wird der Speicher nach jedem Bild ganz frei, und zwischen den Bildern läuft auf oi-o-mat nichts vom Dienst. Die Modelle bleiben, wo sie sind: im Hugging-Face-Cache, den Tonwerk schon benutzt hat (auf der T7).
- **Die Galerie** liegt auch auf oi-o-mat, auf derselben Platte wie die Modelle (`img-o-mat/galerie` ganz oben auf der T7). Gespeichert wird nur, was jemand auf der Seite mit Speichern behält; Bilder für Tonwerk und andere Apps landen dort nie. Ist die T7 nicht angesteckt, sagt die Seite das, statt auf die interne Platte zu schreiben. Ein anderer Ort geht mit `"gallery": "/Volumes/…"` in `worker.json` auf oi-o-mat.
- **Der SSH-Schlüssel** des Dienstes darf auf oi-o-mat nur `worker/launch.py` starten (erzwungener Befehl in `authorized_keys`), keine Shell, keine Weiterleitung.
- **Ein Bild nach dem anderen.** Tonwerk-AI fragt mit Vorrang (`X-Img-Priority: 0`), andere Apps kommen über die LLM-Schranke, die entscheidet, wer welches Modell nutzen darf, und sie zurückhält, solange Tonwerk arbeitet.
- **Lizenzen:** Krea 2 Turbo, FLUX 3 Klein und FLUX.2 Klein 9B prüfen jedes Bild mit Falconsais NSFW-Klassifikator, bevor es herausgeht (Krea-Lizenz §4.2, FLUX-Lizenzen §2.e). Das legt der Katalog fest, keine Anfrage kann es abschalten. Krea 2 und FLUX 3 laufen nur auf einem Mac mit mindestens 32 GB. Jedes Bild trägt in XMP, dass es KI-generiert ist (EU AI Act Art. 50(2)), mit Modell und App.

## Einrichten

1. **Auf oi-o-mat**, in einem Checkout dieses Repositorys:
   ```sh
   ./install-worker.sh
   ```
   Das legt die beiden Python-Umgebungen (mflux 0.23.0, flux3-mlx 0.1.0, so gepinnt wie bisher in Tonwerk) unter `~/Library/Application Support/img-o-mat` an und schreibt `worker.json`. Gibt es Tonwerks Modell-Cache, zeigt `worker.json` darauf, dann wird nichts neu heruntergeladen. „Entfernte Anmeldung“ muss in den Systemeinstellungen an sein.
2. **Ebenfalls auf oi-o-mat:**
   ```sh
   deploy/to-proxmox.sh
   ```
   Das kopiert das Repository auf den Proxmox-Host (ssh-Name `nuc`, sonst `PROXMOX=…` mitgeben) und legt dort beim ersten Mal den Container `img-o-mat` mit Tailscale an. Beim ersten Mal zeigt `tailscale up` einen Link, oder du gibst `TS_AUTHKEY=…` mit. Danach trägt es den Schlüssel des Dienstes auf oi-o-mat ein, überträgt den Hugging-Face-Token und prüft, ob der Dienst oi-o-mat sieht. Für ein Update genügt derselbe Aufruf.
3. Die Seite: **http://img-o-mat:1236/** (über Tailscale).

Der Hugging-Face-Token für Klein 9B und Krea 2 liegt wie bisher in `~/.config/tonwerk/huggingface.token`. Auf oi-o-mat lädt er die Modelle, im Container holt er LoRAs.

### Automatisch ausrollen

Auf oi-o-mat, im selben Checkout (auf `main`, ohne lokale Änderungen):
```sh
deploy/autodeploy.sh --install
```
Ab dann schaut launchd alle fünf Minuten nach, ob `origin/main` neu ist. Wenn ja, wartet es, bis nichts malt, in der Warteschlange steht oder herunterlädt (ein Neustart des Dienstes bräche das ab), spult den Checkout vor, lässt die Tests mit `/usr/bin/python3` laufen und ruft dann `deploy/to-proxmox.sh` auf. Das ruft seinerseits `install-worker.sh` auf, also kommen geänderte Worker-Skripte und neue Runner-Pins mit. Einstellungen für `to-proxmox.sh` (`PROXMOX`, `WORKER`, `LOKI_URL`, …) stehen als `NAME=wert`-Zeilen in `~/.config/img-o-mat/autodeploy.env`.

- Scheitern die Tests, bleibt dieser Stand liegen, bis ein neuer Commit kommt. Scheitert das Ausrollen (Proxmox aus, Tailscale weg), versucht es die nächste Runde wieder. Beides meldet eine Mitteilung auf dem Mac, einmal je Commit.
- `deploy/autodeploy.sh --status` zeigt, was zuletzt ausgerollt wurde, und das Ende des Logs (`~/Library/Logs/img-o-mat/autodeploy.log`). `--uninstall` schaltet es ab.
- Es läuft als dein Benutzer: `git fetch` und `ssh proxmox` müssen ohne Nachfrage gehen (Schlüssel im Schlüsselbund oder ohne Passphrase, Git-Zugang über den Schlüsselbund).

Ohne Proxmox kann der Dienst auch auf einem Mac laufen (`python3 service/server.py`, Einstellungen in `~/.config/img-o-mat/config.json`, `"worker": {"ssh": "marcel@oi-o-mat"}`). Ohne `worker` malt er auf dem eigenen Mac.

### Nachricht, wenn ein Bild fertig ist

Mit [Onkel Bummel](https://github.com/Marcel-B/onkel-bummel) schickt der Dienst eine Nachricht, sobald ein Bild fertig ist, mit den Bildern als Anhang, und eine mit Priorität 4, wenn es fehlschlägt. Abgebrochene Bilder melden nichts.

1. In Onkel Bummel unter Verwaltung → Publish-Tokens einen Token für das Topic `img-o-mat` anlegen.
2. Im Container den Token ablegen: `umask 077; echo 'ob_…' > /etc/img-o-mat/onkel-bummel.token; chown imgomat /etc/img-o-mat/onkel-bummel.token`.
3. In `/etc/img-o-mat/config.json` ergänzen und den Dienst neu starten (`systemctl restart img-o-mat`):
   ```json
   "onkelBummel": {"url": "https://onkel-bummel.<tailnet>.ts.net", "tokenFile": "/etc/img-o-mat/onkel-bummel.token",
                   "topic": "img-o-mat", "click": "http://img-o-mat:1236/", "apps": null}
   ```

`apps` beschränkt die Nachrichten auf diese Apps (`null`: alle, zum Beispiel `["img-o-mat", "3d-o-mat"]`, dann meldet sich Tonwerk nicht). `click` ist die Adresse, die ein Tipp auf die Nachricht öffnet, und darf fehlen. Kommt Onkel Bummel nicht, versucht es der Dienst dreimal und schreibt es dann ins Log; die Warteschlange hält dafür nie an.

## Die Seite

Reines HTML, CSS und JavaScript in einer Datei (`service/ui.html`), ohne Framework, Bibliotheken oder Web-Fonts. Oben stehen vier Bereiche (Malen, Modelle, LoRAs, Galerie; die Adresse merkt sich, welcher offen ist) und ob der Mac, der malt, antwortet. Hell oder dunkel folgt dem System.

- **Malen:** Links Modell, Größe, Anzahl, Seed, die LoRAs, die zum Modell passen (mit eigenem Scrollbereich, wenn es viele sind), und die Warteschlange; rechts oben die Bilder (etwa zwei Drittel der Höhe, scrollen für sich, das neueste zuerst) und fest darunter der Prompt mit Fortschritt; auf dem Handy erst der Prompt, dann die Einstellungen, dann die Bilder; solange gemalt wird, hält ein Platzhalter in der gewählten Größe die Stelle frei. Strg/⌘ + Enter im Prompt malt. Der Knopf oben rechts am Bild (oder ein Klick aufs Bild) öffnet eine große Vorschau mit Herunterladen, Speichern und Löschen; die Pfeiltasten blättern. **Verfeinern** in der Vorschau malt das Bild mit Krea 2 Turbo noch einmal darüber (als Startbild, Nähe 85 %, gleicher Prompt, Seed und Größe; ohne Krea 2 nimmt es Z-Image Turbo): Aufbau, Kleidung und Gesichter bleiben, das Modell bringt mehr Details hinein, etwa Haut nach einem Bild-Edit mit Klein. Die Einstellungen stehen danach in den Malen-Feldern, man kann also mit anderer Nähe gleich noch einmal. Die Bilder liegen nur auf der Seite: Wer neu lädt oder löscht, hat sie nicht mehr, der Dienst behält keine. Speichern legt ein Bild in die Galerie auf der T7 (mit Modell, Seed, Größe und Prompt); dasselbe Bild zweimal wird nur einmal gespeichert. Englisch malt am besten. Tonwerk übersetzt für seine Cover vorher mit Claude, die Seite tut das nicht.
- **Bilder als Vorlage:** „Bild hinzufügen“ unter dem Prompt (oder ein Bild in den Prompt einfügen oder hineinziehen, oder „Als Vorlage“ in der großen Vorschau, auch aus der Galerie) legt bis zu vier Bilder klein über den Prompt. FLUX.2 Klein 4B/9B und FLUX 3 Klein sehen sie an: Der Prompt sagt, was sich ändern soll („make it winter“), oder zeigt mit „image 1“, „image 2“ auf sie. Z-Image Turbo und Krea 2 Turbo malen stattdessen über das erste Bild (img2img), ein Regler sagt, wie nah es am Startbild bleibt (10 bis 90 %). Die Seite verkleinert jedes Bild vorher auf etwa ein Megapixel, und das erste stellt die Größe auf sein Seitenverhältnis. Bilder, die das gewählte Modell nicht nimmt, bleiben liegen und sind als „aus“ markiert.
- **Galerie:** Die gespeicherten Bilder, das neueste zuerst, als kleine Vorschau. Ein Klick öffnet die große Vorschau mit Herunterladen und Löschen (löscht das Bild auf der T7). Darunter steht, womit es gemalt wurde: Prompt (bei FLUX 3 auch, was der Upsampler daraus gemacht hat), Modell, Größe, Seed, die LoRAs mit ihrer Stärke und die Vorlagen als kleine Bilder. Übernehmen trägt das alles wieder in die Malen-Felder ein, auch die Vorlagen; so malt man ein Bild noch einmal oder ändert nur eine Sache. Dasselbe zeigt die Vorschau eines frisch gemalten Bildes. Die kleinen Vorschauen macht `sips` auf oi-o-mat beim Speichern, der Dienst hebt sie auf, damit die Galerie schnell aufgeht.
- **Warteschlange:** Was gerade malt und wer wartet, auch Anfragen anderer Apps.
- **Modelle:** Lizenz, ob kommerziell nutzbar, Speicher, ob heruntergeladen, und ein Knopf zum Herunterladen (15 bis 35 GB auf die T7; er prüft vorher den Platz). Nie von selbst.
- **LoRAs:** Oben wählst du ein Modell. Darunter stehen die LoRAs, die für dieses Modell schon auf der Platte liegen (Name, Modell, Trigger und Stärke ändern, löschen), und die, die du von Hugging Face laden kannst (bisher nur Krea 2 Turbo, Krea hat für die anderen Modelle keine offiziellen). Eigene Dateien lädst du unten hoch; das Modell erkennt der Dienst meist aus der Datei, sonst gilt das oben gewählte.

## Schnittstelle

OpenAI-kompatibel, wo OpenAI etwas hat, also gehen auch dessen Clients. Fehler haben OpenAIs Form: `{"error": {"message", "type": "img_o_mat", "code"}}`.

| Aufruf | Wirkung |
| --- | --- |
| `GET /v1/models` | die Modelle mit `available`/`unavailable` (Grund), `downloaded`, `memoryGb`, `downloadGb`, `license`, `commercial`, `contentFilter`, `loras`, `references`, `startImage`, `lettering`; `?fresh=1` fragt oi-o-mat neu |
| `POST /v1/models/<id>/download` | Download auf oi-o-mat starten (202); Stand unter `download` im Modell |
| `POST /v1/images/generations` | malen, siehe unten |
| `DELETE /v1/jobs/<id>` | ein Bild abbrechen (die Kennung steht in `X-Img-Job`) |
| `HEAD`/`PUT /v1/files/<sha256>` | Referenzbild (oder LoRA außerhalb der Bibliothek) einmal hochladen, danach per Hash nennen |
| `POST /v1/files` | ein Bild (bis 32 MB) hochladen, ohne den Hash vorher zu kennen; Antwort `{"sha256"}`, `201` neu, `200` schon da (so lädt die Seite ihre Vorlagen hoch) |
| `GET /v1/loras`, `PUT`/`PATCH`/`DELETE /v1/loras/<name>`, `POST /v1/loras/fetch` | die LoRA-Bibliothek; `PUT` nimmt die Datei als Inhalt, `label`, `model`, `trigger`, `strength` als Query |
| `GET /v1/busy` | was gerade malt: `busy`, `app`, `model`, `memoryGb`, `since`, `waiting`; billig, Tonwerk fragt es, bevor es den Speicher belegt |
| `GET /v1/gallery` | die gespeicherten Bilder (`name`, `model`, `seed`, `size`, `prompt`, `app`, `savedAt`, `bytes`) und `disk`; `503`, wenn die Platte fehlt |
| `POST /v1/gallery` | ein Bild behalten: `{"b64_json", "model", "seed", "size", "prompt", "paintedPrompt", "loras": [{"name", "label", "strength"}], "references": [{"file", "role", "strength"}]}`; die Vorlagen nimmt der Dienst aus seinen hochgeladenen Dateien und legt sie mit auf die T7; `201` neu, `200` schon da, Antwort `{"name", "created"}` |
| `GET`/`DELETE /v1/gallery/<name>` | das Bild als JPEG (`?thumb=1` klein) oder löschen; eine Vorlage geht mit dem letzten Bild, das sie nennt |
| `GET /v1/gallery/ref/<sha256>` | eine Vorlage eines gespeicherten Bildes als JPEG |
| `GET /status` | oi-o-mat (Speicher, Platz, Token), was malt, was wartet, die letzten Bilder, Downloads |

```sh
curl http://img-o-mat:1236/v1/images/generations -H 'content-type: application/json' -H 'Authorization: Bearer meine-app' -d '{
  "model": "klein-4b", "prompt": "a lighthouse at dusk, oil painting", "size": "1024x1024", "n": 1, "seed": 42,
  "loras": [{"name": "krea-neondrip", "strength": 0.8}],
  "references": [{"file": "<sha256>", "role": "style"}]
}'
```

- `model` ist die Kennung aus `/v1/models` (`klein-4b`, `klein-9b`, `z-image-turbo`, `krea-2-turbo`, `flux3-klein`).
- `size` ist `BreitexHöhe`, je 256 bis 2048 und durch 16 teilbar. `n` geht von 1 bis 4, die Seeds werden hochgezählt.
- `loras` (bis 3, nur mflux-Modelle) nennt LoRAs der Bibliothek beim `name` oder hochgeladene beim `file`. Eine LoRA passt nur zu ihrem Modell.
- `references`: Bilder, die FLUX.2 Klein und FLUX 3 ansehen (bis 4, Rolle `style`, `figure` oder `place`). Wofür „image 1“, „image 2“ stehen, schreibt die App selbst in den Prompt. Z-Image und Krea 2 malen stattdessen über ein Startbild (`role: "start"`, `strength` 0.1 bis 0.9).
- `"notify": false` schickt für dieses Bild keine Nachricht an Onkel Bummel (siehe unten).
- `creator` (bis 60 Zeichen) steht im XMP-Vermerk. Ohne Angabe gilt der App-Name aus `Authorization: Bearer <app>` bzw. `X-Llm-App`.
- Die Antwort ist OpenAIs `{"created", "data": [{"b64_json", "revised_prompt", "seed", …}]}`. `revised_prompt` ist bei FLUX 3 der Text, den sein Upsampler daraus gemacht hat.
- Mit `"stream": true` kommen Server-Sent Events: `started` (`id`, `seed`), `queued` (`place`), `progress` (`stage`: `sending`, `loading`, `painting`, `checking`; `fraction`), `image` (je Bild, wie ein Eintrag von `data`), dann `done` oder `error`. Wer die Verbindung schließt, bricht das Bild ab.
- `422` heißt: Der Inhaltsfilter hat das Bild zurückgehalten. `501`: Das Modell läuft auf oi-o-mat nicht (Grund in der Meldung). `502`: oi-o-mat ist nicht erreichbar.

Der Dienst hat keine Anmeldung, wie Tonwerk und die Schranke. Er gehört ins Heimnetz und Tailscale. Apps sollen über die Schranke kommen, damit Freigabe und Rücksicht auf Tonwerk gelten.

## Überwachung

`GET /metrics` liefert Zahlen für Prometheus, ohne Anmeldung und ohne Inhalte. Das sind:
- Bilder je App, Modell und Ergebnis (`done`, `failed`, `filtered`, `cancelled`)
- Mal- und Wartezeit
- die Warteschlange und welches Modell gerade malt
- Fehler auf dem Weg zu oi-o-mat (`ssh`, `runner`, `transfer`) und ob oi-o-mat zuletzt antwortete (geprüft alle 2 Minuten)
- Downloads, belegter Platz für Referenzbilder und LoRAs, Anfragen je Route und Speicher und CPU des Dienstes

Als App zählen nur `tonwerk-ai`, `3d-o-mat` und `img-o-mat` (die Seite). Alle anderen erscheinen als `other`, damit ein fremder Schlüssel keine neuen Reihen anlegt. Die Liste lässt sich in `config.json` unter `"metrics": {"apps": [...]}` ändern. Das Label heißt `client`, wie bei der LLM-Schranke, weil Prometheus das Ziel schon mit `app=img-o-mat` kennzeichnet.

Die Logzeilen gehen zusätzlich an Loki, wenn in `config.json` `"loki": {"url": "http://monitoring:3100", "host": "proxmox"}` steht. Die Labels sind `app=img-o-mat`, `host`, `level` und `source` (`painter`, `worker`, `downloads`, `loras`, `files`, `http`, `config`). `deploy/to-proxmox.sh` trägt das mit `LOKI_URL=http://monitoring:3100` ein. Ist Loki weg, werden die Zeilen verworfen und gezählt. Das Journal bleibt, wie es ist. Prompts stehen in keiner Zeile, weder dort noch in Loki.

Das Dashboard und das Ziel liegen im Monitoring-Stack (Tonwerk-AI, `deploy/monitoring/`).

## Entwickeln

Python 3.9 oder neuer, ohne Pakete. `python3 -m unittest discover tests` spielt den ganzen Weg gegen nachgestellte Maler (`tests/fake`) durch, lokal und über einen nachgestellten `ssh`, der wie der erzwungene Befehl arbeitet. Die echten Maler (`worker/paint.py`, `fetch.py`) brauchen einen Mac mit Apple Silicon.

## Was getestet ist

Stand 2026-11-08: Verfeinern lief in Chromium mit den nachgestellten Malern (Klein 9B, dann Krea 2 mit dem Bild als Startbild, Nähe 85 %, gleicher Seed und Größe). Ob 85 % mit echtem Krea 2 die richtige Nähe ist, zeigt erst der Versuch auf oi-o-mat.

Stand 2026-11-08: Prompt, LoRAs, Seed und Vorlagen in der Galerie laufen in den Tests (lokal) und einmal in Chromium mit den nachgestellten Malern (speichern, Detailansicht, Übernehmen). Auf oi-o-mat mit der T7 noch nicht.

Stand 2026-11-08: Vorlagen auf der Seite liefen in Chromium mit den nachgestellten Malern (zwei Bilder für Klein 4B, ein Bild aus der Vorschau, Startbild für Z-Image), und `POST /v1/files` in den Tests. Ob Klein (mflux' Edit-Pipeline), FLUX 3 und das Startbild bei Z-Image und Krea 2 auf oi-o-mat wirklich malen, zeigt erst das erste echte Bild; dieser Weg stammt aus Tonwerk und lief hier noch nie mit echten Modellen.

Stand 2026-11-05: Die Galerie läuft in den Tests lokal und über den nachgestellten `ssh` (speichern, doppelt speichern, auflisten, ausliefern, löschen, fehlende Platte), und die Seite einmal in Chromium mit den nachgestellten Malern (Speichern in der Vorschau, Galerie, Löschen). Auf oi-o-mat mit der echten T7 und `sips`, und im Proxmox-Container, ist sie noch nicht gelaufen.

Stand 2026-11-05: Die Nachricht an Onkel Bummel ist nur gegen einen nachgestellten Onkel Bummel in den Tests gelaufen, noch nicht gegen den echten.

Stand 2026-11-05: `deploy/autodeploy.sh` läuft in den Tests gegen ein Wegwerf-Repository und einen nachgestellten Dienst (einmal je Commit, Warten beim Malen und Herunterladen, lokale Änderungen, scheiternde Tests und scheiterndes Ausrollen). launchd, die Mitteilungen und der echte Weg nach Proxmox sind auf oi-o-mat noch nicht gelaufen.

Stand 2026-10-30: `/metrics` und der Versand an Loki laufen gegen die nachgestellten Maler und einen kleinen nachgestellten Loki. Die Ausgabe besteht `promtool check metrics`, und die Tests laufen mit Python 3.9 und 3.13. Im Proxmox-Container ist das noch nicht gelaufen.

Stand 2026-10-28: Die Tests laufen in einem Linux-Container. Einmal lief es dort auch über einen echten `sshd` mit dem erzwungenen Befehl, mit LoRA-Übertragung, Abbruch (der entfernte Prozess endet) und Download, alles mit den nachgestellten Malern. `worker/paint.py` ist Tonwerks bewährtes Malskript mit kleinen Änderungen (Bild über stdout, Dateien per Hash). Auf oi-o-mat, in Proxmox und mit echten Modellen ist es noch nicht gelaufen. Ob MLX in einer SSH-Sitzung auf oi-o-mat die GPU bekommt, zeigt erst das erste Bild.

## Lizenz

PolyForm Noncommercial 1.0.0 (`LICENSE`, `NOTICE`). Die Modelle haben ihre eigenen Lizenzen (siehe Seite und `service/catalogue.py`).
