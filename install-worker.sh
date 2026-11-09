#!/bin/sh
# Sets up the Mac that paints (oi-o-mat): four Python environments with the runners (mflux, flux3-mlx, ltx-2-mlx
# and minimax-h3-mlx for video clips), the worker scripts and
# worker.json, under ~/Library/Application Support/img-o-mat. The service (on Proxmox, deploy/to-proxmox.sh) starts
# them over SSH; nothing runs here between pictures.
#
#   ./install-worker.sh                      install, or move the runners to the versions pinned here
#   ./install-worker.sh --key "<public key>" also let that key (the service's) run worker/launch.py, and nothing else
#
# The models stay where they are: if Tonwerk-AI's cache exists (~/Library/Application Support/YuE UI/images/models,
# on oi-o-mat a link to the T7), worker.json points there, so nothing is downloaded again. Klein 9B and Krea 2 are
# gated: their licences accepted on huggingface.co and a read token in ~/.config/tonwerk/huggingface.token.
set -eu

ROOT="$HOME/Library/Application Support/img-o-mat"
HERE="$(cd "$(dirname "$0")" && pwd)"
TONWERK_CACHE="$HOME/Library/Application Support/YuE UI/images/models"
KEY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --key) KEY="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unbekannt: $1" >&2; exit 2 ;;
  esac
done

[ -x /usr/bin/python3 ] && /usr/bin/python3 -c 'import sys' 2>/dev/null || {
  echo "/usr/bin/python3 fehlt: xcode-select --install" >&2; exit 1; }

# Pinned as Tonwerk-AI had them (deploy/install-images.sh there, 2026-10-22): 0.23 brought Krea 2.
PACKAGE="mflux==0.23.0"
FLUX3="flux3-mlx[pup]==0.1.0"
# Pillow 13 imports numbers._IntegralLike at run time, which no released Python has (seen 2026-10-21).
PILLOW="pillow<13"
# The content filter (Falconsai's classifier through transformers) needs torch; flux3-mlx does not bring it.
TORCH="torch==2.14.0"
# ltx-2-mlx is not on PyPI: its two packages from the repository at a fixed commit (0.18.0, 2026-11-01). It writes the
# mp4 with Homebrew's ffmpeg.
LTX_COMMIT="747b30ef10e92a011e85b2aa0f6c4ee86ced5218"
LTX_GIT="git+https://github.com/dgrauet/ltx-2-mlx@$LTX_COMMIT"
# The H3 port has no installable package; retain its CLI and modules at this exact revision.
MINIMAX_COMMIT="b2f7e4d2b7861cefe68b75e4b59ab81cc4e7c318"
# Sampling-only fork with one-block residency; encoder/VAEs retain PipeNetwork's pinned implementation.
MINIMAX_STREAM_COMMIT="3fcc9f8c557cf353400b1acf737b985e524a509f"

BASE=""
if ! command -v uv >/dev/null 2>&1; then
  for candidate in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      BASE="$(command -v "$candidate")"
      break
    fi
  done
  [ -n "$BASE" ] || { echo "Kein Python 3.10 oder neuer (brew install uv)." >&2; exit 1; }
fi

install() {
  python="$1"
  shift
  env="$(dirname "$(dirname "$python")")"
  echo "Umgebung: $env"
  if [ -z "$BASE" ]; then
    [ -x "$python" ] || uv venv --python 3.12 "$env"
    uv pip install --python "$python" "$@"
  else
    [ -x "$python" ] || "$BASE" -m venv "$env"
    "$python" -m pip install --quiet --upgrade pip
    "$python" -m pip install "$@"
  fi
}

mkdir -p "$ROOT/worker" "$ROOT/files"
install "$ROOT/env/bin/python" "$PACKAGE" "$PILLOW"
"$ROOT/env/bin/python" -c 'import importlib.metadata as m; from mflux.models.flux2.variants import Flux2Klein; from mflux.models.krea2 import Krea2; from mflux.models.z_image import ZImage; import huggingface_hub; print("mflux", m.version("mflux"))'
install "$ROOT/flux3/bin/python" "$FLUX3" "$PILLOW" "$TORCH"
"$ROOT/flux3/bin/python" -c 'import importlib.metadata as m; from flux3_mlx import Flux3Klein; import mlx_vlm, torch; from transformers import pipeline; print("flux3-mlx", m.version("flux3-mlx"))'
# ltx-2-mlx needs Python 3.11; uv brings 3.12, a Python found here may be older.
if [ -z "$BASE" ] || "$BASE" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
  install "$ROOT/ltx/bin/python" \
    "ltx-core-mlx @ $LTX_GIT#subdirectory=packages/ltx-core-mlx" \
    "ltx-pipelines-mlx @ $LTX_GIT#subdirectory=packages/ltx-pipelines-mlx" "$PILLOW"
  "$ROOT/ltx/bin/python" -c 'import importlib.metadata as m; from ltx_pipelines_mlx.distilled import DistilledPipeline; import huggingface_hub; print("ltx-2-mlx", m.version("ltx-pipelines-mlx"))'
  [ -x "$ROOT/ltx/bin/ltx-2-mlx" ] || { echo "ltx-2-mlx fehlt in $ROOT/ltx/bin" >&2; exit 1; }
  command -v ffmpeg >/dev/null 2>&1 || [ -x /opt/homebrew/bin/ffmpeg ] || echo "Hinweis: ffmpeg fehlt (brew install ffmpeg); ohne macht ltx-2-mlx kein Video."
else
  echo "Hinweis: ltx-2-mlx braucht Python 3.11 oder neuer (brew install uv); Videos gehen sonst nicht."
fi

# No weights are downloaded by installation; the page fetches the quant and the upstream components explicitly.
command -v git >/dev/null 2>&1 || { echo "git fehlt (xcode-select --install)." >&2; exit 1; }
if [ ! -d "$ROOT/minimax-h3-mlx/.git" ]; then
  git clone https://github.com/PipeNetwork/minimax-h3-mlx.git "$ROOT/minimax-h3-mlx"
fi
git -C "$ROOT/minimax-h3-mlx" fetch origin "$MINIMAX_COMMIT"
git -C "$ROOT/minimax-h3-mlx" checkout --detach "$MINIMAX_COMMIT"
install "$ROOT/minimax/bin/python" -r "$ROOT/minimax-h3-mlx/requirements.txt" "mlx==0.32.4" "mlx-vlm==0.7.12" "$PILLOW"
PYTHONPATH="$ROOT/minimax-h3-mlx" "$ROOT/minimax/bin/python" -c 'from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline; from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel; from transformers import AutoTokenizer, AutoProcessor; from PIL import Image; import huggingface_hub; print("minimax-h3-mlx bereit")'

if [ ! -d "$ROOT/minimax-h3-streaming/.git" ]; then
  git clone https://github.com/Argus-AiTeam/minimax-h3-mac.git "$ROOT/minimax-h3-streaming"
fi
git -C "$ROOT/minimax-h3-streaming" fetch origin "$MINIMAX_STREAM_COMMIT"
git -C "$ROOT/minimax-h3-streaming" checkout --detach "$MINIMAX_STREAM_COMMIT"
PYTHONPATH="$ROOT/minimax-h3-streaming" "$ROOT/minimax/bin/python" -c 'from minimax_h3_mlx.streaming import load_streaming_dit; from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline; print("MiniMax-Blockstreaming bereit")'

cp "$HERE"/worker/*.py "$ROOT/worker/"
if [ ! -f "$ROOT/worker.json" ]; then
  CACHE="/Volumes/T7/img-o-mat/models"
  if [ -e "$TONWERK_CACHE" ] && /usr/bin/python3 -c 'import os, sys; sys.exit(not os.path.realpath(sys.argv[1]).startswith("/Volumes/T7/"))' "$TONWERK_CACHE"; then
    CACHE="$TONWERK_CACHE"
  fi
  /usr/bin/python3 -c 'import json, sys; json.dump({"modelCache": sys.argv[1]}, open(sys.argv[2], "w"), indent=2)' "$CACHE" "$ROOT/worker.json"
  echo "Modelle in $CACHE"
fi

if [ -n "$KEY" ]; then
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  touch "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"
  # One line, tagged, replaced on every run: the key may run launch.py only, without a terminal or forwarding.
  grep -v ' img-o-mat-service$' "$HOME/.ssh/authorized_keys" > "$HOME/.ssh/authorized_keys.tmp" || true
  set -- $KEY
  printf 'restrict,command="/usr/bin/python3 \\"%s/worker/launch.py\\" --ssh" %s %s img-o-mat-service\n' "$ROOT" "$1" "$2" >> "$HOME/.ssh/authorized_keys.tmp"
  mv "$HOME/.ssh/authorized_keys.tmp" "$HOME/.ssh/authorized_keys"
  echo "Schlüssel des Dienstes eingetragen."
fi

IMG_O_MAT_WORKER_ROOT="$ROOT" /usr/bin/python3 "$ROOT/worker/launch.py" probe <<'JSON'
{"repos": ["black-forest-labs/FLUX.2-klein-4B"]}
JSON
echo
echo "Fertig. Prüfe, dass Entfernte Anmeldung (Systemeinstellungen > Allgemein > Teilen) an ist."
