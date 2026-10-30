# img-o-mat

Die Bildmodelle von oi-o-mat als Dienst für alle Apps: FLUX.2 Klein 4B und 9B, Z-Image Turbo, Krea 2 Turbo und FLUX 3 Klein, mit Bild-LoRAs, Referenzbildern und einer Seite, auf der man Modelle und LoRAs verwaltet und direkt Prompts malt. Bisher hat Tonwerk-AI diese Modelle selbst gestartet; jetzt kann jede App sie nutzen, auch Tonwerk.

## Aufbau

```
 Apps ──► LLM-Schranke (oi-o-mat:1235) ──┐
                                         ├──► img-o-mat (Proxmox-Container, :1236) ──ssh──► oi-o-mat malt
 Tonwerk-AI (direkt, mit Vorrang) ───────┘        Seite, Warteschlange, LoRAs              Modelle auf der T7
```

- **Der Dienst** läuft in einem kleinen Debian-Container auf Proxmox (512 MB). Dort liegen die Seite, die Warteschlange, die LoRA-Bibliothek und hochgeladene Referenzbilder. Er ist reines Python ohne Pakete.
- **Gemalt wird auf oi-o-mat.** Für jedes Bild startet der Dienst per SSH einen Prozess mit mflux oder flux3-mlx, der das Modell lädt, malt und wieder geht. So wird der Speicher nach jedem Bild ganz frei, und zwischen den Bildern läuft auf oi-o-mat nichts vom Dienst. Die Modelle bleiben, wo sie sind: im Hugging-Face-Cache, den Tonwerk schon benutzt hat (auf der T7).
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
   Das kopiert das Repository auf den Proxmox-Host (ssh-Name `proxmox`) und legt dort beim ersten Mal den Container `img-o-mat` mit Tailscale an. Beim ersten Mal zeigt `tailscale up` einen Link, oder du gibst `TS_AUTHKEY=…` mit. Danach trägt es den Schlüssel des Dienstes auf oi-o-mat ein, überträgt den Hugging-Face-Token und prüft, ob der Dienst oi-o-mat sieht. Für ein Update genügt derselbe Aufruf.
3. Die Seite: **http://img-o-mat:1236/** (über Tailscale).

Der Hugging-Face-Token für Klein 9B und Krea 2 liegt wie bisher in `~/.config/tonwerk/huggingface.token`. Auf oi-o-mat lädt er die Modelle, im Container holt er LoRAs.

Ohne Proxmox kann der Dienst auch auf einem Mac laufen (`python3 service/server.py`, Einstellungen in `~/.config/img-o-mat/config.json`, `"worker": {"ssh": "marcel@oi-o-mat"}`). Ohne `worker` malt er auf dem eigenen Mac.

## Die Seite

- **Malen:** Modell, Größe, Anzahl, Seed, Prompt und die LoRAs, die zum Modell passen. Fortschritt, Ergebnisse zum Herunterladen. Englisch malt am besten. Tonwerk übersetzt für seine Cover vorher mit Claude, die Seite tut das nicht.
- **Warteschlange:** Was gerade malt und wer wartet, auch Anfragen anderer Apps.
- **Modelle:** Lizenz, ob kommerziell nutzbar, Speicher, ob heruntergeladen, und ein Knopf zum Herunterladen (15 bis 35 GB auf die T7; er prüft vorher den Platz). Nie von selbst.
- **LoRAs:** Hochladen, Name, Modell, Trigger und Stärke ändern, löschen, Krea-2-Stil-LoRAs von Hugging Face holen. Das Modell erkennt der Dienst meist aus der Datei.

## Schnittstelle

OpenAI-kompatibel, wo OpenAI etwas hat, also gehen auch dessen Clients. Fehler haben OpenAIs Form: `{"error": {"message", "type": "img_o_mat", "code"}}`.

| Aufruf | Wirkung |
| --- | --- |
| `GET /v1/models` | die Modelle mit `available`/`unavailable` (Grund), `downloaded`, `memoryGb`, `downloadGb`, `license`, `commercial`, `contentFilter`, `loras`, `references`, `startImage`, `lettering`; `?fresh=1` fragt oi-o-mat neu |
| `POST /v1/models/<id>/download` | Download auf oi-o-mat starten (202); Stand unter `download` im Modell |
| `POST /v1/images/generations` | malen, siehe unten |
| `DELETE /v1/jobs/<id>` | ein Bild abbrechen (die Kennung steht in `X-Img-Job`) |
| `HEAD`/`PUT /v1/files/<sha256>` | Referenzbild (oder LoRA außerhalb der Bibliothek) einmal hochladen, danach per Hash nennen |
| `GET /v1/loras`, `PUT`/`PATCH`/`DELETE /v1/loras/<name>`, `POST /v1/loras/fetch` | die LoRA-Bibliothek; `PUT` nimmt die Datei als Inhalt, `label`, `model`, `trigger`, `strength` als Query |
| `GET /v1/busy` | was gerade malt: `busy`, `app`, `model`, `memoryGb`, `since`, `waiting`; billig, Tonwerk fragt es, bevor es den Speicher belegt |
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

Als App zählen nur `tonwerk-ai`, `3d-o-mat` und `img-o-mat` (die Seite). Alle anderen erscheinen als `other`, damit ein fremder Schlüssel keine neuen Reihen anlegt. Die Liste lässt sich in `config.json` unter `"metrics": {"apps": [...]}` ändern.

Die Logzeilen gehen zusätzlich an Loki, wenn in `config.json` `"loki": {"url": "http://monitoring:3100", "host": "proxmox"}` steht. Die Labels sind `app=img-o-mat`, `host`, `level` und `source` (`painter`, `worker`, `downloads`, `loras`, `files`, `http`, `config`). `deploy/to-proxmox.sh` trägt das mit `LOKI_URL=http://monitoring:3100` ein. Ist Loki weg, werden die Zeilen verworfen und gezählt. Das Journal bleibt, wie es ist. Prompts stehen in keiner Zeile, weder dort noch in Loki.

Das Dashboard und das Ziel liegen im Monitoring-Stack (Tonwerk-AI, `deploy/monitoring/`).

## Entwickeln

Python 3.9 oder neuer, ohne Pakete. `python3 -m unittest discover tests` spielt den ganzen Weg gegen nachgestellte Maler (`tests/fake`) durch, lokal und über einen nachgestellten `ssh`, der wie der erzwungene Befehl arbeitet. Die echten Maler (`worker/paint.py`, `fetch.py`) brauchen einen Mac mit Apple Silicon.

## Was getestet ist

Stand 2026-10-30: `/metrics` und der Versand an Loki laufen gegen die nachgestellten Maler und einen kleinen nachgestellten Loki. Die Ausgabe besteht `promtool check metrics`, und die Tests laufen mit Python 3.9 und 3.13. Im Proxmox-Container ist das noch nicht gelaufen.

Stand 2026-10-28: Die Tests laufen in einem Linux-Container. Einmal lief es dort auch über einen echten `sshd` mit dem erzwungenen Befehl, mit LoRA-Übertragung, Abbruch (der entfernte Prozess endet) und Download, alles mit den nachgestellten Malern. `worker/paint.py` ist Tonwerks bewährtes Malskript mit kleinen Änderungen (Bild über stdout, Dateien per Hash). Auf oi-o-mat, in Proxmox und mit echten Modellen ist es noch nicht gelaufen. Ob MLX in einer SSH-Sitzung auf oi-o-mat die GPU bekommt, zeigt erst das erste Bild.

## Lizenz

PolyForm Noncommercial 1.0.0 (`LICENSE`, `NOTICE`). Die Modelle haben ihre eigenen Lizenzen (siehe Seite und `service/catalogue.py`).
