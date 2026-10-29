#!/bin/sh
# Sets up the Mac that paints (oi-o-mat): two Python environments with the runners, the worker scripts and
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

cp "$HERE"/worker/*.py "$ROOT/worker/"
if [ ! -f "$ROOT/worker.json" ]; then
  CACHE="$ROOT/models"
  [ -e "$TONWERK_CACHE" ] && CACHE="$TONWERK_CACHE"
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
