# MiniMax H3 auf 32 GB

Stand: 9. November 2026. Im Branch implementiert, noch kein vollständiger Lauf mit den großen Gewichten auf oi-o-mat. Die Grenze von 32 GB ist ein experimentelles Ziel, kein gemessenes Funktionsversprechen.

## Ergebnis der Recherche

Das Problem ist die gleichzeitige Belegung durch mehrere Komponenten, nicht allein die Zahl „4 Bit“. [PipeNetworks Modellkarte](https://huggingface.co/pipenetwork/minimax-h3-mlx-4bit) nennt rund 25,3 GB Transformerdateien und 11,5 GB residente Transformergewichte nach dem Vorberechnen und Entfernen der AdaLN-Projektionen. Ein zusätzlicher, dauerhaft geladener Qwen-Encoder und beide VAEs machen diese Ersparnis teilweise wieder zunichte.

| Ansatz | Primärquelle | Einordnung für img-o-mat |
| --- | --- | --- |
| Komponenten nacheinander laden | [MLX-Serve-Studio: Speicher und Anforderungen](https://github.com/SawadaShusaku/minimax-h3-mlx-studio#requirements), [MLX-Serve: Belegung nach Phasen](https://github.com/ddalcu/mlx-serve/issues/126) | Die Quellen beschreiben getrennte Encoder-/Transformer-Belegung; das ist der Ansatz dieses Branches. Angaben anderer Runner sind keine Messungen unseres Python-Runners. |
| Gewichte blockweise streamen | [Argus: MiniMax auf 24 GB](https://github.com/Argus-AiTeam/minimax-h3-mac/blob/main/README_EN.md) | Der Anbieter berichtet einen Lauf auf einem M4 Pro mit 24 GB durch Streaming. Andere Lade- und Ausführungspfade, teils andere Gewichte; möglicher nächster Schritt, wenn unser Ladepeak zu groß bleibt. |
| Textencoder auf 2 Bit reduzieren | [H3-spezifischer 2-Bit-Encoder](https://huggingface.co/antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder) | Rund 9,6 statt 15,8 GB laut Modellkarte, ebenfalls getrennte Phasen. Der Anbieter nennt möglichen Verlust bei Prompttreue. Vorerst nicht eingebunden: 4 Bit und die vorhandenen Gewichte bleiben erhalten. |

## Implementierter Weg

Bei weniger als 64 GB meldet der Videokatalog `lowRam: true`. Der Dienst erlaubt für MiniMax dann 5 Sekunden bei 768 × 768. Größere Anfragen werden vor der Warteschlange abgewiesen. Ein Anfrageparameter kann diese hardwareabhängigen Grenzen nicht ausschalten.

`worker/minimax_low_ram.py` startet jede Phase erst, nachdem der vorherige Prozess beendet ist:

1. Qwen in 4 Bit: Prompt und gegebenenfalls Bildinformationen kodieren.
2. Bei Start-/Endbildern: deren Videokonditionierung mit dem originalen VAE und dessen Seed-42-Verfahren berechnen.
3. Den vorhandenen PipeNetwork-Transformer laden und den originalen Sampler ausführen. Encoder und VAEs sind hier nicht geladen. AdaLN wird wie bisher gecacht und entfernt.
4. Nur den Video-VAE laden und die Videolatenten dekodieren.
5. Nur den Audio-VAE laden und die Audiolatenten dekodieren.
6. Die Ergebnisse mit ffmpeg zusammenführen.

Die Zwischenstände liegen beim temporären Clip. Normale Fertigstellung und Fehler räumen sie auf. Alle Kindprozesse erben die Prozessgruppe des Workers; die bestehende Abbruchfunktion beendet damit auch die aktive Phase. Bei einem harten Prozessabbruch können wie beim bisherigen Worker temporäre Dateien zurückbleiben.

Der MLX-Cache ist pro Phase auf 256 MiB begrenzt. `mx.set_memory_limit(24 GiB)` setzt eine lokale Speicher-Richtgröße; sie garantiert keine harte Obergrenze und umfasst nicht den gesamten Prozess. Eine Erhöhung globaler macOS-Wired-Memory-Grenzen erfolgt nicht. `peakMemoryGb` meldet den größten tatsächlich beobachteten MLX-Phasenpeak in GiB.

## Was geprüft ist und was fehlt

Die Diensttests prüfen Profilwahl, Parametergrenzen und unveränderte Modelltrennung. Kleine echte MLX-Tests vergleichen Video- und Audiolatenten des ursprünglichen Samplers mit denen nach der Phasentrennung, jeweils bei gleichem Seed, auch mit Start-/Endkonditionierung. Weitere Tests prüfen, dass die Samplingphase keine VAEs und die Decoderphase keinen Transformer lädt. Reihenfolge, Fehlerabbruch und normale Bereinigung sind getestet.

Es fehlen ein vollständiger 32-GB-Lauf, dessen Spitzenbedarf einschließlich macOS, mögliche Swap-Nutzung, Laufzeit und eine Sicht-/Hörprüfung des Ergebnisses. Es wurden keine großen Gewichte automatisch heruntergeladen und keine Änderungen auf oi-o-mat oder Proxmox ausgerollt. Der erste Versuch sollte mit 5 Sekunden, 768 × 768, 8 Schritten und zunächst ohne Bilder erfolgen; danach Start-/Endbilder prüfen. Reicht das nicht, ist blockweises Streaming der nächste technisch passende Schritt.
