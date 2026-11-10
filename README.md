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
   Das legt die vier Python-Umgebungen (mflux 0.23.0, flux3-mlx 0.1.0, so gepinnt wie bisher in Tonwerk, und ltx-2-mlx 0.18.0 für Videos, aus GitHub auf einem festen Commit, braucht Python 3.11 über `uv` und `brew install ffmpeg`, sowie minimax-h3-mlx auf dem festen Commit `b2f7e4d2b7861cefe68b75e4b59ab81cc4e7c318` mit eigener Umgebung, MLX 0.32.4 und mlx-vlm 0.7.12) unter `~/Library/Application Support/img-o-mat` an und schreibt `worker.json`. Gibt es Tonwerks Modell-Cache auf der T7, zeigt die neue `worker.json` darauf, dann wird nichts neu heruntergeladen. Andernfalls ist bei einer neuen Installation `/Volumes/T7/img-o-mat/models` das Ziel. Eine vorhandene `worker.json` wird nicht überschrieben. „Entfernte Anmeldung“ muss in den Systemeinstellungen an sein.
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

- **Malen:** Links Modell, Größe, Anzahl, Seed, die LoRAs, die zum Modell passen (mit eigenem Scrollbereich, wenn es viele sind), und die Warteschlange; rechts oben die Bilder (etwa zwei Drittel der Höhe, scrollen für sich, das neueste zuerst) und fest darunter der Prompt mit Fortschritt; auf dem Handy erst der Prompt, dann die Einstellungen, dann die Bilder; solange gemalt wird, hält ein Platzhalter in der gewählten Größe die Stelle frei. Strg/⌘ + Enter im Prompt malt. „Groß“ über dem Prompt öffnet ihn in einem großen Fenster (auf dem Handy bildschirmfüllend) mit Zeichenzähler, „Fertig“ und „Malen“; Esc oder „Klein“ bringt ihn ins kleine Feld zurück, der Text bleibt derselbe. Während gemalt wird, heißt der Knopf **Einreihen**: Jeder Druck stellt einen weiteren Auftrag mit dem Prompt, Modell, den LoRAs und Vorlagen von diesem Moment in die Reihe, abgearbeitet wird nacheinander. Der Auftrag, der malt, und der nächste liegen schon beim Dienst (so macht der Mac keine Pause, und das Ausrollen wartet), die übrigen warten auf der Seite. Das × auf einem Platzhalter nimmt seinen Auftrag heraus, Abbrechen den, der gerade dran ist; ein Fehler bleibt als Karte stehen, die Reihe läuft weiter. Wer die Seite schließt, bricht die Reihe ab. Der Knopf oben rechts am Bild (oder ein Klick aufs Bild) öffnet eine große Vorschau mit Herunterladen, Speichern und Löschen; die Pfeiltasten blättern. **Verfeinern** in der Vorschau malt das Bild mit Krea 2 Turbo noch einmal darüber (als Startbild, Nähe 85 %, gleicher Prompt, Seed und Größe; ohne Krea 2 nimmt es Z-Image Turbo): Aufbau, Kleidung und Gesichter bleiben, das Modell bringt mehr Details hinein, etwa Haut nach einem Bild-Edit mit Klein. Die Einstellungen stehen danach in den Malen-Feldern, man kann also mit anderer Nähe gleich noch einmal. Die Bilder liegen nur auf der Seite: Wer neu lädt oder löscht, hat sie nicht mehr, der Dienst behält keine. Speichern legt ein Bild in die Galerie auf der T7 (mit Modell, Seed, Größe, Prompt und wie lange es gedauert hat: Malen und Modell laden, wie der Mac es misst); dasselbe Bild zweimal wird nur einmal gespeichert. Bei Krea 2 Turbo steht unter dem Prompt eine Zeile **Negativprompt** (was nicht ins Bild soll), im großen Fenster unter dem Prompt zwei Zeilen; bei den anderen Modellen ist sie ausgeblendet, weil sie dort nichts bewirkt, der Text bleibt aber stehen. Der Prompt-Bereich wächst dafür nach oben, die Bilder darüber werden etwas kleiner. Die Vorschau zeigt ihn als „Nicht im Bild“, Speichern und Übernehmen nehmen ihn mit. Englisch malt am besten. Tonwerk übersetzt für seine Cover vorher mit Claude, die Seite tut das nicht.
- **Bilder als Vorlage:** „Bild hinzufügen“ unter dem Prompt (oder ein Bild in den Prompt einfügen oder hineinziehen, oder „Als Vorlage“ in der großen Vorschau, auch aus der Galerie) legt bis zu vier Bilder klein über den Prompt. FLUX.2 Klein 4B/9B und FLUX 3 Klein sehen sie an: Der Prompt sagt, was sich ändern soll („make it winter“), oder zeigt mit „image 1“, „image 2“ auf sie. Z-Image Turbo und Krea 2 Turbo malen stattdessen über das erste Bild (img2img), ein Regler sagt, wie nah es am Startbild bleibt (10 bis 90 %). Die Seite verkleinert jedes Bild vorher auf etwa ein Megapixel, und das erste stellt die Größe auf sein Seitenverhältnis. Bilder, die das gewählte Modell nicht nimmt, bleiben liegen und sind als „aus“ markiert.
- **Galerie:** Die gespeicherten Bilder, das neueste zuerst, als kleine Vorschau. Ein Klick öffnet die große Vorschau mit Herunterladen und Löschen (löscht das Bild auf der T7). Darunter steht, womit es gemalt wurde: Prompt (bei FLUX 3 auch, was der Upsampler daraus gemacht hat), Modell, Größe, Seed, die LoRAs mit ihrer Stärke und die Vorlagen als kleine Bilder. Übernehmen trägt das alles wieder in die Malen-Felder ein, auch die Vorlagen; so malt man ein Bild noch einmal oder ändert nur eine Sache. Dasselbe zeigt die Vorschau eines frisch gemalten Bildes. Die kleinen Vorschauen macht `sips` auf oi-o-mat beim Speichern, der Dienst hebt sie auf, damit die Galerie schnell aufgeht.
- **Warteschlange:** Was gerade malt und wer wartet, auch Anfragen anderer Apps, dahinter die Aufträge, die noch auf der Seite warten.
- **Modelle:** Lizenz, ob kommerziell nutzbar, Speicher, ob heruntergeladen, und ein Knopf zum Herunterladen (15 bis 35 GB auf die T7; er prüft vorher den Platz). Nie von selbst.
- **Video:** Ein kurzer Clip (2, 4 oder 5 Sekunden, 24 Bilder pro Sekunde, wahlweise mit Ton) mit LTX-2.3 über ltx-2-mlx. Dazu kommen ein Prompt, was passiert, und bis zu zwei Bilder: Das Bild am **Anfang** ist genau das erste Bild des Clips, auf das Bild am **Ende** läuft der Clip zu (ltx-2-mlx' `--image … last … guide`). So endet ein Intro zum Beispiel genau auf einer Spielerkarte. Bilder kommen über „Bild wählen“, durch Hineinziehen oder mit „Für Video“ aus der großen Vorschau (auch aus der Galerie). „Schnell“ ist LTX' destillierter Weg (8 + 3 Schritte), „gut“ der zweistufige mit 30 Schritten. Ein Clip läuft durch dieselbe Warteschlange wie die Bilder, und auf einem Mac mit 32 GB streamt ltx-2-mlx die Gewichte von der Platte (`--low-ram`). Der fertige Clip bleibt nur auf der Seite; Herunterladen behält ihn. Seine Karte sagt, wie lange er gefilmt wurde; der Knopf oben rechts öffnet ihn groß mit Modell, Länge, Größe, Seed, Qualität, Ton, Dauer und den Bildern am Anfang und Ende, und die Pfeile (oder die Pfeiltasten) blättern zum nächsten Clip. Er trägt in seinen MP4-Metadaten den Hinweis, dass er KI-generiert ist.
- **MiniMax H3 (4 bit):** Im Reiter Video zusätzlich auswählbar, sobald Runner und Modelle vorhanden sind. Text zu Video und Start-/Endbilder über den [MLX-Port von PipeNetwork](https://github.com/PipeNetwork/minimax-h3-mlx); synchroner Ton wird mitgeneriert und bei ausgeschaltetem „mit Ton“ anschließend entfernt. „Schnell“ nutzt 8, „gut“ 16 Denoising-Schritte (9 bzw. 17 Sigma-Gitterpunkte). Länge 5, 10 oder 15 Sekunden, 24 fps, durch das VAE-Raster etwas aufgerundet (z. B. 124 Frames für 5 Sekunden); 768 × 768, 1344 × 768 oder Hochformat. Die API erlaubt auch Zwischenlängen. Die kurze Bildkante bleibt bei den trainierten 768 Pixeln. Bildstärke wird nicht unterstützt; Start-/Endbild-Konditionierung ist im Port implementiert, vom Anbieter aber noch nicht durchgehend erprobt.
  **Powered by MiniMax H3.** Die [4-Bit-Gewichte](https://huggingface.co/pipenetwork/minimax-h3-mlx-4bit) enthalten nur den Transformer (25,3 GB). Hinzu kommt der [H3-spezifische Qwen3-VL-32B in 4 Bit](https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit): nur `text_encoder.safetensors` (15,8 GB), Quantisierungskonfiguration und Lizenzhinweise aus diesem Paket. Aus `MiniMaxAI/MiniMax-H3` kommen weiterhin die originale Encoder-Konfiguration, Tokenizer, Processor und beide VAEs (zusammen etwa 11 GB), aber **keine unquantisierten Textencoder- oder Transformergewichte**. Zusammen rund **53 GB Download**. `worker/minimax_generate.py` lädt die vorquantisierten Sprach- und Bildencoder-Gewichte direkt im MLX-Affine-Format (4 Bit, Gruppengröße 64). Er behält die H3-spezifische Promptaufbereitung und die Ausgabe vor der finalen Normalisierung aus Schicht 50 bei; er ersetzt kein Modell durch einen beliebigen anderen Qwen. Token- und Positions-Embeddings bleiben wie im Paket unquantisiert, die Bildencoder-Faltung wird ins MLX-Layout umgeordnet. Der Runner nutzt dafür die geprüften Versionen MLX 0.32.4 und mlx-vlm 0.7.12.
  Ab **32 GB Systemspeicher** gibt es nun einen **experimentellen Modus mit getrennten Phasen**, bei weniger als 64 GB automatisch: zunächst **5 Sekunden bei 768 × 768**, wahlweise mit Start-/Endbild und Ton. Textencoder, gegebenenfalls Bild-VAE, Transformer, Video-Decoder und Audio-Decoder laufen nacheinander in eigenen Prozessen. Die Samplingphase nutzt den auf Commit `3fcc9f8c557cf353400b1acf737b985e524a509f` festgelegten [Argus-Streaming-Fork](https://github.com/Argus-AiTeam/minimax-h3-mac): nur statische Gewichte und ein wiederverwendbarer Transformerblock sind gleichzeitig resident; auch AdaLN wird blockweise vorberechnet. Die anderen Phasen verwenden weiterhin den festgelegten PipeNetwork-Runner. Beide Quellcodeverzeichnisse werden bei `install-worker.sh` installiert, ohne Modellgewichte zu laden. Zwischen ihnen werden nur Konditionierung und Latenten auf der Platte ausgetauscht; die großen Gewichte verlassen den Speicher mit dem jeweiligen Prozess. Die Samplingphase und die anderen Phasen laufen mit getrennten Python-Prozessen, damit sich die namensgleichen Runner-Pakete nicht vermischen. Die Decoder bleiben beim bisherigen Runner; es gibt keine neue Quantisierung, kein Überspringen von Transformerblöcken und keinen approximativen Block-Cache. Der lokale MLX-Puffer-Cache ist je Phase auf 256 MiB begrenzt; die Speicher-Richtgröße auf 24 GiB. Diese Richtgröße ist **keine harte Prozessgrenze**. Es werden keine globalen macOS-Speichereinstellungen geändert.
  **Initial herunterladen:** Unter **Modelle → Videos → MiniMax H3 (4 bit)** auf **„Auf T7 herunterladen (53 GB)“** klicken. Der eine Knopf lädt Transformer, den 4-Bit-Textencoder und die benötigten Originalkomponenten; Stand und Fehler erscheinen auf der Modellkarte. Ist die T7 nicht eingehängt oder zeigt der Cache auf die interne SSD, ist der Knopf gesperrt. Auch die API und der Worker lehnen diesen Download ab, bevor ein Cache-Ordner angelegt wird. Der Worker prüft vor jeder Repository-Dateigruppe erneut den T7-Speicherort. Hub-, Xet- und Assets-Cache werden auf dieselbe Platte festgelegt; geerbte interne Hugging-Face-Cachepfade werden überschrieben. Bei einer bestehenden internen Cache-Konfiguration muss `modelCache` in der Worker-Konfiguration auf einen Ordner der T7 zeigen; vorhandene T7-Pfade können bleiben. Es wird nichts automatisch verschoben und kein Modelldownload von allein gestartet.
  **T7 und interne SSD:** Streaming liest die Modell-Shards direkt aus dem eingestellten Modell-Cache; es legt keine zweite Modellkopie auf der internen SSD an. `modelCache` in `worker.json` muss deshalb wie vorgesehen auf die T7 zeigen. Intern entstehen nur die Job-Zwischenstände und der temporäre Clip, standardmäßig unter `~/Library/Application Support/img-o-mat/video-work`. Dieser Scratch-Pfad kann mit `videoWork` geändert werden. Nach Erfolg, Fehler und Abbruch löscht der Launcher den Job-Ordner; nach einem Absturz werden verwaiste verwaltete Ordner bei der nächsten Worker-Abfrage oder dem nächsten Videojob aufgeräumt. Aktive Jobs und fremde Verzeichnisse werden nicht entfernt. Dauerhaft bleiben nur die installierten Programme/Python-Umgebungen, keine generierten Job-Dateien. Der größte rohe Zwischenstand eines 124-Frame-Clips bei 768 × 768 ist rechnerisch rund 209 MiB RGB-Daten, zuzüglich Latenten, Audio und MP4.
  Die Angabe von 26 GB Speicher im kleinen Profil ist eine **Schätzung**, kein gemessener vollständiger Lauf. Die Transformerdateien bleiben rund 25,3 GB groß, werden im kleinen Profil aber blockweise verarbeitet. Es wird dort kein vollständiger Transformer mehr geladen. Ab 64 GB bleibt das bisherige Profil mit bis zu 15 Sekunden und 1344 × 768 verfügbar (44 GB geschätzt). Der erste vollständige Lauf auf oi-o-mat steht aus; vorher lässt sich der Betrieb auf 32 GB nicht garantieren. Der Worker meldet den größten gemessenen MLX-Phasenbedarf als `peakMemoryGb` im Ergebnis. Das ist die MLX-Allokation in GiB, nicht der gesamte Prozess- oder Systemspeicher. Hintergründe und alternative Runner stehen unter [32-GB-Recherche](docs/minimax-32gb.md).
  Die dichte Attention braucht weiterhin viel Zeit; auch für fünf Sekunden sind Stunden möglich. Die Wirkung der Encoder-Quantisierung auf die Videoqualität ist noch nicht lokal geprüft. `minimaxTimeoutMinutes` setzt ein eigenes Zeitlimit (Standard 1440 Minuten), unabhängig von LTX/Bildern. Es gelten die MiniMax H3 Community License, die Umsatzgrenze von 20 Mio. US-Dollar und deren territoriale Einschränkungen.

- **Videos** (unter Modelle): LTX-2.3 in 8 Bit (81 GB mit dem Textencoder Gemma 3 12B) und Wan 2.2 TI2V 5B in 8 Bit (für rapid-mlx, 20 GB, auf dem Stand, den rapid-mlx 1.0.2 festlegt; damit gibt es noch keine Clips). Sie stehen weder im Malen-Formular noch in `GET /v1/models`, Tonwerk sieht sie also nicht. LTX und Wan brauchen einen Mac mit 32 GB. MiniMax H3 ergänzt die Liste mit den oben beschriebenen Anforderungen.
- **LoRAs:** Oben wählst du ein Modell. Darunter stehen die LoRAs, die für dieses Modell schon auf der Platte liegen (Name, Modell, Trigger und Stärke ändern, löschen), und die, die du von Hugging Face laden kannst (bisher nur Krea 2 Turbo, Krea hat für die anderen Modelle keine offiziellen). Eigene Dateien lädst du unten hoch; das Modell erkennt der Dienst meist aus der Datei, sonst gilt das oben gewählte.

## Schnittstelle

OpenAI-kompatibel, wo OpenAI etwas hat, also gehen auch dessen Clients. Fehler haben OpenAIs Form: `{"error": {"message", "type": "img_o_mat", "code"}}`.

| Aufruf | Wirkung |
| --- | --- |
| `GET /v1/models` | die Modelle mit `available`/`unavailable` (Grund), `downloaded`, `memoryGb`, `downloadGb`, `license`, `commercial`, `contentFilter`, `loras`, `references`, `startImage`, `negativePrompt`, `lettering`; `?fresh=1` fragt oi-o-mat neu |
| `GET /v1/models?kind=video` | die Videomodelle (`kind: "video"`, `paints`, `ready`/`notReady`, `downloaded`, `lowRam`) |
| `POST /v1/videos/generations` | ein Clip: `model` (`ltx-2.3` oder `minimax-h3-mlx-4bit`), `prompt`, `size` (256 bis 1280, durch 64 teilbar, Standard 768x768), `frames` (8·k+1 bis 121, Standard 97 = 4 s), `quality` (`fast`/`high`), `audio`, `seed`, `images` (`[{"file": "<sha256>", "at": "first" \| "last"}]`, aus `/v1/files`), `stream`; MiniMax nutzt stattdessen `duration` (5–15 s, Standard 5) ohne `frames`, `size` mit kurzer Kante 768 (maximal 1344, durch 32 teilbar) und Bildstärke 1; unter 64 GB nur 5 s bei 768x768 (automatischer Phasenmodus); Antwort `data[0].b64_mp4`, mit `stream` als SSE (`progress`, `video`, `done`) |
| `POST /v1/models/<id>/download` | Download auf oi-o-mat starten (202), auch für ein Videomodell; Stand unter `download` im Modell |
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
- `negative_prompt` (bis 2000 Zeichen, nicht OpenAIs): was nicht ins Bild soll. Nur Krea 2 Turbo nimmt ihn (`negativePrompt` in `/v1/models`); es malt dann mit echtem CFG (Guidance 2 statt 1) und braucht etwa doppelt so lange. Die anderen Modelle sind destilliert und malen ohne CFG, wo ein Negativprompt nichts bewirkt: mflux setzt die Guidance bei Z-Image Turbo auf 0 und gibt FLUX.2 Klein einen festen leeren, flux3-mlx lehnt Guidance bei FLUX 3 Klein ab. Dort antwortet der Dienst mit `400`, statt ihn stillschweigend zu verwerfen; ein leerer zählt nicht.
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

Hängt ein Bild, obwohl auf oi-o-mat nichts mehr malt, schreibt `systemctl kill -s USR1 img-o-mat` im Container, wo jeder Thread des Dienstes gerade steht, ins Journal (`journalctl -u img-o-mat`). Das vor einem Neustart ausführen, sonst ist die Spur weg.

Das Dashboard und das Ziel liegen im Monitoring-Stack (Tonwerk-AI, `deploy/monitoring/`).

## Entwickeln

Python 3.9 oder neuer, ohne Pakete. `python3 -m unittest discover tests` spielt den ganzen Weg gegen nachgestellte Maler (`tests/fake`) durch, lokal und über einen nachgestellten `ssh`, der wie der erzwungene Befehl arbeitet. Die echten Maler (`worker/paint.py`, `fetch.py`) brauchen einen Mac mit Apple Silicon.

## Was getestet ist

Stand 2026-11-09 (MiniMax-Branch): Modellkatalog, selektiver Downloadauftrag, Speichergrenze, API-Prüfung, Übergabe an die eigene Worker-Umgebung, Start-/Endbilder, Fortschrittsparser und Entfernen des Tons sind mit nachgestellten Runnern bzw. isolierten Aufrufen getestet. Der Runner ist auf den oben genannten Quellcode-Commit festgelegt; Modellgewichte werden ausschließlich über den Downloadknopf geladen. Keine echte MiniMax-Installation oder Generierung auf oi-o-mat und keine Bereitstellung auf Proxmox durchgeführt. Die Speichergrenzen beider Profile müssen mit echten vollständigen Läufen überprüft werden. Der 4-Bit-Loader wurde zusätzlich mit kleinen echten MLX-Gewichten geprüft (gepacktes uint32 bleibt erhalten, numerisch identischer quantisierter Vorwärtslauf, Bildencoder-Layout, Fehler bei unvollständigen Gewichten). Alle 1780 Tensor-Namen und Formen des veröffentlichten Encoder-Dateikopfs wurden mit den MLX-Modulen abgeglichen; die großen Encodergewichte wurden dafür nicht geladen. Die optionalen MLX-Tests laufen mit dem Python der MiniMax-Umgebung über `IMG_O_MAT_MINIMAX_SOURCE="<Pfad zum installierten minimax-h3-mlx>" IMG_O_MAT_MINIMAX_STREAMING_SOURCE="<Pfad zum installierten minimax-h3-streaming>" python -m unittest discover -s tests -p "test_minimax*.py"`; ohne MLX bzw. Runner-Quellcode überspringt die Dienst-Testsuite die betreffenden Prüfungen. Dazu kommen Vergleiche des ursprünglichen Samplers mit den Phasengrenzen (Text sowie Start-/Endbilder, Video- und Audiolatenten bei gleichem Seed) und Prüfungen, dass jede Decoderphase nur ihre eigenen Gewichte lädt. Die Tests verwenden kleine Daten und ersetzen die großen Gewichte; sie messen nicht den realen H3-Spitzenbedarf. Ein zusätzlicher Vergleich erstellt mit dem originalen PipeNetwork-Runner einen kleinen 4-Bit-Checkpoint samt 8-Bit-AdaLN und lädt ihn im separaten Argus-Prozess: Video- und Audiovorhersagen sowie die Latenten der gesamten Samplingphase sind bitgleich zur ursprünglichen Pipeline, bei einem residenten Block-Slot für drei Modellblöcke. Der Pfad `minimaxStreamingSource` kann bei Bedarf in `worker.json` angepasst werden.

Stand 2026-11-10: Blättern zwischen Clips und die Dauer bei Bildern und Clips liefen in Chromium mit den nachgestellten Malern (Bild malen, speichern, in der Galerie wieder ansehen; zwei Clips groß ansehen, mit Pfeiltasten und Knöpfen blättern, löschen); die Dauer in der Galerie läuft in den Tests. Auf oi-o-mat und im Container auf Proxmox noch nicht.

Stand 2026-11-09: Clips laufen in den Tests gegen einen nachgestellten ltx-2-mlx (lokal und über den nachgestellten `ssh`, Bild am Anfang und am Ende, Prüfung der Werte, Fehler), der Aufruf, den `worker/video.py` baut, ist getestet, und der Reiter Video lief in Chromium auf Desktop- und Handybreite. Den echten ltx-2-mlx (Installation, Fortschritt aus seinen Balken, Speicher auf 32 GB, Dauer) hat noch kein Clip auf oi-o-mat gesehen.

Stand 2026-11-09: Die Videomodelle laufen in den Tests (nicht in `GET /v1/models`, Download mit dem festgelegten Stand für Wan) und einmal in Chromium mit den nachgestellten Malern (Abschnitt Videos, Platzprüfung bei LTX). Ob die echten Repositories mit diesen Namen und Größen auf die T7 kommen, zeigt erst der erste Download auf oi-o-mat; die Größen stehen so auf Hugging Face, der Speicher ist geschätzt.

Stand 2026-11-09: Der Negativprompt lief in Chromium mit den nachgestellten Malern (Feld nur bei Krea 2, Anfrage, Vorschau) und in den Tests bis zu mflux' Aufruf. Ob Guidance 2 bei Krea 2 Turbo mit acht Schritten gut aussieht und wie lange es auf oi-o-mat dauert, zeigt erst das erste echte Bild; auf Proxmox ist es noch nicht ausgerollt.

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
