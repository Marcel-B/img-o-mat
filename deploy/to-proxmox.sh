#!/bin/sh
# On oi-o-mat, from this checkout: brings img-o-mat's service to its Proxmox container and connects the two. It
# copies the repository to the Proxmox host, runs deploy/proxmox-lxc.sh there (first time: creates the container), lets
# the service's key run worker/launch.py here (install-worker.sh --key), gives the container the Hugging Face token
# for fetching LoRAs and checks that the service sees this Mac.
#
#   deploy/to-proxmox.sh            set up or update
#
# Settings through the environment: PROXMOX (ssh name of the Proxmox host, "proxmox"), WORKER (how the container
# reaches this Mac, $USER@<this Mac's name>), plus those of proxmox-lxc.sh (TS_AUTHKEY, CTID, LOKI_URL, ...).
# install-worker.sh must have run here once (the environments take a while; it is not repeated here).
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROXMOX="${PROXMOX:-proxmox}"
WORKER="${WORKER:-$(id -un)@$(scutil --get LocalHostName 2>/dev/null | tr '[:upper:]' '[:lower:]' || hostname -s)}"
TOKEN="$HOME/.config/tonwerk/huggingface.token"
[ -f "$HOME/.config/img-o-mat/huggingface.token" ] && TOKEN="$HOME/.config/img-o-mat/huggingface.token"

[ -f "$HOME/Library/Application Support/img-o-mat/worker.json" ] || {
  echo "Erst ./install-worker.sh auf diesem Mac ausführen." >&2; exit 1; }

echo "Kopiere das Repository nach $PROXMOX:/root/img-o-mat ..."
tar -C "$HERE" -czf - service worker deploy | ssh "$PROXMOX" 'rm -rf /root/img-o-mat && mkdir -p /root/img-o-mat && tar -C /root/img-o-mat -xzf -'
OUT="$(ssh "$PROXMOX" "WORKER='$WORKER' TS_AUTHKEY='${TS_AUTHKEY:-}' CTID='${CTID:-}' LOKI_URL='${LOKI_URL:-}' /root/img-o-mat/deploy/proxmox-lxc.sh" | tr -d '\r')"
CTID="$(printf '%s\n' "$OUT" | sed -n 's/^CTID=//p' | tail -n 1)"
KEY="$(printf '%s\n' "$OUT" | grep '^ssh-ed25519 ' | tail -n 1)"
[ -n "$CTID" ] && [ -n "$KEY" ] || { echo "proxmox-lxc.sh lieferte keinen Schlüssel:" >&2; printf '%s\n' "$OUT" >&2; exit 1; }

"$HERE/install-worker.sh" --key "$KEY" >/dev/null
echo "Schlüssel des Dienstes darf hier worker/launch.py starten."

if [ -f "$TOKEN" ]; then
  ssh "$PROXMOX" "pct exec $CTID -- sh -c 'umask 077; cat > /var/lib/img-o-mat/huggingface.token; chown imgomat /var/lib/img-o-mat/huggingface.token'" < "$TOKEN"
  echo "Hugging-Face-Token übertragen."
fi

echo "Prüfe, ob der Dienst diesen Mac erreicht ..."
for _ in 1 2 3 4 5 6; do
  if curl -fsS "http://img-o-mat:1236/v1/models?fresh=1" 2>/dev/null | /usr/bin/python3 -c '
import json, sys
models = json.load(sys.stdin)["data"]
for m in models:
    state = "bereit" if m["available"] and m["downloaded"] else "nicht geladen" if m["available"] else m["unavailable"]
    print("  " + m["label"] + ": " + state)
sys.exit(0 if any(m["available"] for m in models) else 1)'; then
    echo "Fertig. Seite: http://img-o-mat:1236/"
    exit 0
  fi
  sleep 5
done
echo "Der Dienst erreicht diesen Mac nicht; http://img-o-mat:1236/ zeigt oben den Grund." >&2
exit 1
