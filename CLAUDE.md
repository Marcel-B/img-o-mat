# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

img-o-mat: the image models (FLUX.2 Klein 4B/9B, Z-Image Turbo, Krea 2 Turbo, FLUX 3 Klein) as a service for every app on Marcel's machines, with a page to manage models and LoRAs and paint by hand. Moved out of Tonwerk-AI (`Marcel-B/Tonwerk-AI`, `src/YueUI.Api/Images/`) on 2026-10-28; Tonwerk is meant to become a client. README.md (German) is the user documentation.

## Layout

- `service/` runs in a Debian LXC on Proxmox (`deploy/proxmox-lxc.sh`, systemd unit `img-o-mat`, config `/etc/img-o-mat/config.json`, data `/var/lib/img-o-mat`), Python standard library only, 3.9 or newer, port 1236. `server.py` (HTTP API, OpenAI-compatible where OpenAI has one, SSE for progress), `painter.py` (one picture at a time, priority then arrival; Tonwerk-AI asks with `X-Img-Priority: 0`), `catalogue.py` (models with the same ids as Tonwerk's `ImageModel.Defaults`; `contentFilter` and `minSystemGb` come from here and no request can change them), `worker_link.py` (SSH to the Mac that paints, or local without `worker`), `files.py` (uploads by SHA-256), `loras.py` (library, moved from Tonwerk's `ImageLoras.cs`), `downloads.py`, `ui.html` (the page, German, no web fonts or libraries).
- `worker/` is copied to oi-o-mat by `install-worker.sh` (`~/Library/Application Support/img-o-mat/worker`). `launch.py` is the only thing the service's SSH key may run (forced command `--ssh`; it reads the words after the first of `SSH_ORIGINAL_COMMAND`) and the only entry point either way: it starts `paint.py`/`fetch.py` with the runner's Python from `worker.json` and kills them when its stdin ends (the service closes it to cancel; over SSH nothing else reaches the remote process). `probe.py` and `launch.py` must run on macOS's `/usr/bin/python3` (3.9): keep `from __future__ import annotations` and no newer syntax. `paint.py` is Tonwerk's `yueui_image.py` with the picture returned as base64 on stdout and files named by hash.
- Events between worker and service are lines starting with `IMGOMAT ` followed by JSON; everything else is the runner's own output, kept for the error message (an `Error: …` line wins).

## Rules

- The licences ask for the content filter on Krea 2, FLUX 3 and Klein 9B; keep it in the catalogue and never let a request switch it off. Every picture carries the XMP AI note (EU AI Act Art. 50(2)) with model and app.
- Model ids are stored by clients (Tonwerk's pictures name them): never rename one. Runner pins in `install-worker.sh` change only through a commit.
- Nothing is downloaded by itself: models only by the page's button or `POST /v1/models/<id>/download`.
- Tests (`python3 -m unittest discover tests`) come with every change; `tests/fake` stands in for the runners and `ssh`.
- Commit messages and branch names are German; code and comments English; UI strings and README German. Comments explain why.
- License PolyForm Noncommercial 1.0.0; never bring in GPL/AGPL code.
- Say what was and was not tested on oi-o-mat and Proxmox.
