# MiniMax H3 auf 32 GB

Stand: 9. November 2026. Im Branch implementiert, noch kein vollständiger Lauf mit den großen Gewichten auf oi-o-mat. Die Grenze von 32 GB ist ein experimentelles Ziel, kein gemessenes Funktionsversprechen.

## Ergebnis der Recherche

Das Problem ist die gleichzeitige Belegung durch mehrere Komponenten, nicht allein die Zahl „4 Bit“. [PipeNetworks Modellkarte](https://huggingface.co/pipenetwork/minimax-h3-mlx-4bit) nennt rund 25,3 GB Transformerdateien und 11,5 GB residente Transformergewichte nach dem Vorberechnen und Entfernen der AdaLN-Projektionen. Ein zusätzlicher, dauerhaft geladener Qwen-Encoder und beide VAEs machen diese Ersparnis teilweise wieder zunichte.

| Ansatz | Primärquelle | Einordnung für img-o-mat |
| --- | --- | --- |
| Komponenten nacheinander laden | [MLX-Serve-Studio: Speicher und Anforderungen](https://github.com/SawadaShusaku/minimax-h3-mlx-studio#requirements), [MLX-Serve: Belegung nach Phasen](https://github.com/ddalcu/mlx-serve/issues/126) | Die Quellen beschreiben getrennte Encoder-/Transformer-Belegung; die Komponententrennung ist Teil dieses Branches. Angaben anderer Runner sind keine Messungen unseres Python-Runners. |
| Gewichte blockweise streamen | [Argus: MiniMax auf 24 GB](https://github.com/Argus-AiTeam/minimax-h3-mac/blob/main/README_EN.md) | Der Anbieter berichtet einen Lauf auf einem M4 Pro mit 24 GB durch Streaming. Dieser Branch nutzt nun den festgelegten Argus-Streaming-Code ausschließlich in der Samplingphase, mit den vorhandenen PipeNetwork-4-Bit-Gewichten. Der veröffentlichte 24-GB-Lauf verwendet andere Gewichte und Turbo; seine Messwerte sind nicht auf unsere Variante übertragen. |
| Textencoder auf 2 Bit reduzieren | [H3-spezifischer 2-Bit-Encoder](https://huggingface.co/antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder) | Rund 9,6 statt 15,8 GB laut Modellkarte, ebenfalls getrennte Phasen. Der Anbieter nennt möglichen Verlust bei Prompttreue. Vorerst nicht eingebunden: 4 Bit und die vorhandenen Gewichte bleiben erhalten. |

## Implementierter Weg

Bei weniger als 64 GB meldet der Videokatalog `lowRam: true`. Der Dienst erlaubt für MiniMax dann 5 Sekunden bei 768 × 768. Größere Anfragen werden vor der Warteschlange abgewiesen. Ein Anfrageparameter kann diese hardwareabhängigen Grenzen nicht ausschalten.

`worker/minimax_low_ram.py` startet jede Phase erst, nachdem der vorherige Prozess beendet ist:

1. Qwen in 4 Bit: Prompt und gegebenenfalls Bildinformationen kodieren.
2. Bei Start-/Endbildern: deren Videokonditionierung mit dem originalen VAE und dessen Seed-42-Verfahren berechnen.
3. Den vorhandenen PipeNetwork-4-Bit-Transformer über den Argus-Streaming-Fork ausführen (Commit `3fcc9f8c557cf353400b1acf737b985e524a509f`). Nur statische Gewichte und ein Block-Slot sind resident. Die übrigen Blöcke werden der Reihe nach nachgeladen und materialisiert, bevor der Slot neu belegt wird. Auch AdaLN wird blockweise vorberechnet. Encoder und VAEs sind hier nicht geladen; es gibt keinen Rückfall auf vollständiges Laden und keinen approximativen Block-Cache.
4. Nur den Video-VAE laden und die Videolatenten dekodieren.
5. Nur den Audio-VAE laden und die Audiolatenten dekodieren.
6. Die Ergebnisse mit ffmpeg zusammenführen.

Der Installer behält zwei Quellcodeverzeichnisse bei: `minimax-h3-mlx` für Encoder und Decoder sowie `minimax-h3-streaming` für den Sampler. Die Sampling-Unterprozesse importieren ausschließlich den Fork, die anderen Phasen ausschließlich PipeNetwork. Ein fehlender Streaming-Runner führt vor dem ersten Modellladen zu einer Fehlermeldung. Die Vorabprüfung verlangt beide Installationen.

Die Gewichte werden direkt von den Shards im eingestellten `modelCache` gelesen. Es gibt kein vorheriges Kopieren des Modells von der T7 auf die interne SSD und keinen zweiten persistenten Modell-Cache. Voraussetzung ist, dass `modelCache` auf die T7 zeigt. Das Betriebssystem kann gelesene Seiten im RAM zwischenspeichern und bei Druck eigenen Auslagerungsspeicher verwenden; der Worker erstellt dafür keine eigene SSD-Modellkopie.

Der native Launcher legt vor dem Videojob einen verwalteten Scratch-Ordner an, standardmäßig unter `~/Library/Application Support/img-o-mat/video-work` (überschreibbar durch `videoWork` in `worker.json`). Darin liegen die Zwischenstände und der temporäre Clip. Der Launcher bleibt beim Abbrechen des Modellprozesses am Leben und löscht den Ordner nach dessen Ende, auch nach Fehler oder SIGKILL der Worker-Prozessgruppe. Alle Kindprozesse erben diese Gruppe, sodass Abbrechen auch die aktive Streamingphase beendet. Nach einem System-/Launcher-Absturz werden Ordner mit nicht mehr existierendem Besitzerprozess bei der nächsten Worker-Abfrage bzw. vor dem nächsten Videojob entfernt. Aktive, unbekannte oder fremde Ordner sowie Symlinks werden nicht rekursiv gelöscht. Programmcode und Python-Umgebungen bleiben installiert; Job-Daten bleiben nicht als dauerhafter interner Cache bestehen.

Der MLX-Cache ist pro Phase auf 256 MiB begrenzt. `mx.set_memory_limit(24 GiB)` setzt eine lokale Speicher-Richtgröße; sie garantiert keine harte Obergrenze und umfasst nicht den gesamten Prozess. Eine Erhöhung globaler macOS-Wired-Memory-Grenzen erfolgt nicht. `peakMemoryGb` meldet den größten tatsächlich beobachteten MLX-Phasenpeak in GiB.

## Was geprüft ist und was fehlt

Die Diensttests prüfen Profilwahl, Parametergrenzen und unveränderte Modelltrennung. Kleine echte MLX-Tests vergleichen Video- und Audiolatenten des ursprünglichen Samplers mit denen nach der Phasentrennung, jeweils bei gleichem Seed, auch mit Start-/Endkonditionierung. Weitere Tests prüfen, dass die Samplingphase keine VAEs und die Decoderphase keinen Transformer lädt. Reihenfolge, Fehlerabbruch und normale Bereinigung sind getestet. Ein zusätzlicher echter MLX-Test schreibt einen kleinen gemischt quantisierten 4-/8-Bit-Checkpoint mit PipeNetwork und lädt ihn in einem separaten Argus-Prozess. Die Video- und Audiovorhersagen sowie die Video-/Audiolatenten der vollständigen kleinen Samplingphase sind bitgleich zur ursprünglichen Pipeline. Der Test verlangt genau einen residenten Block-Slot, obwohl das Testmodell drei Blöcke hat. Auch die Streaming-Tests des Forks (4 Bit und BF16, ein und zwei Slots) wurden ausgeführt.

Es fehlen ein vollständiger 32-GB-Lauf, dessen Spitzenbedarf einschließlich macOS, mögliche Swap-Nutzung, Laufzeit und eine Sicht-/Hörprüfung des Ergebnisses. Es wurden keine großen Gewichte automatisch heruntergeladen und keine Änderungen auf oi-o-mat oder Proxmox ausgerollt. Der erste Versuch sollte mit 5 Sekunden, 768 × 768, 8 Schritten und zunächst ohne Bilder erfolgen; danach Start-/Endbilder prüfen. Reicht das nicht, muss die problematische Phase anhand des Peaks weiter eingegrenzt werden, insbesondere Encoder- und VAE-Spitzenbedarf. Blockweises Transformer-Streaming ist bereits implementiert. Die derzeitigen 5-Sekunden-/768-Quadrat-Grenzen bleiben bis zur echten Messung bestehen.
