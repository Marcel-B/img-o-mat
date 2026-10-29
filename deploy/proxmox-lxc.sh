#!/bin/bash
# Sets up img-o-mat's service in a Debian LXC on the Proxmox host: Python, Tailscale, a user of its own with an SSH key
# for oi-o-mat, and a systemd unit. Run it as root on the Proxmox host from a copy of this repository; usually
# deploy/to-proxmox.sh on oi-o-mat does that, copies the key to oi-o-mat and checks the result.
#
# Settings through the environment, all optional:
#   WORKER       who paints, as ssh reaches it from the container (marcel@oi-o-mat)
#   TS_AUTHKEY   a Tailscale auth key; without one, tailscale up prints a link to sign the container in
#   CT_HOSTNAME  container name, also its Tailscale name (img-o-mat)
#   CTID         container id (the next free one)
#   STORAGE      storage for the root disk (local-lvm), TEMPLATE_STORAGE for the Debian template (local)
#   BRIDGE       network bridge (vmbr0), MEMORY in MB (512), CORES (1), DISK in GB (16; LoRAs and uploads)
#
# Running it again finds the container by its name, copies the service anew and restarts it; config, key and LoRAs
# stay. The last line it prints is the service's public key.
set -euo pipefail

CT_HOSTNAME="${CT_HOSTNAME:-img-o-mat}"
STORAGE="${STORAGE:-local-lvm}"
TEMPLATE_STORAGE="${TEMPLATE_STORAGE:-local}"
BRIDGE="${BRIDGE:-vmbr0}"
MEMORY="${MEMORY:-512}"
CORES="${CORES:-1}"
DISK="${DISK:-16}"
TS_AUTHKEY="${TS_AUTHKEY:-}"
WORKER="${WORKER:-marcel@oi-o-mat}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "Unbekannt: $1" >&2; exit 2 ;;
esac

command -v pct >/dev/null || { echo "pct fehlt: auf dem Proxmox-Host ausführen." >&2; exit 1; }
[ "$(id -u)" = 0 ] || { echo "Als root ausführen." >&2; exit 1; }
[ -f "$HERE/service/server.py" ] || { echo "service/server.py fehlt neben diesem Skript." >&2; exit 1; }

in_ct() { pct exec "$CTID" -- bash -c "$1"; }

CTID="${CTID:-$(pct list | awk -v name="$CT_HOSTNAME" 'NR > 1 && $NF == name { print $1; exit }')}"
if [ -n "$CTID" ] && pct status "$CTID" >/dev/null 2>&1; then
  echo "Nehme Container $CTID ($CT_HOSTNAME)." >&2
else
  CTID="${CTID:-$(pvesh get /cluster/nextid)}"
  pveam update >/dev/null
  TEMPLATE="$(pveam available --section system | awk '$2 ~ /^debian-12-standard_/ { print $2 }' | sort -V | tail -n 1)"
  [ -n "$TEMPLATE" ] || { echo "Keine Debian-12-Vorlage bei pveam." >&2; exit 1; }
  pveam list "$TEMPLATE_STORAGE" | grep -q "$TEMPLATE" || pveam download "$TEMPLATE_STORAGE" "$TEMPLATE" >&2
  echo "Lege Container $CTID ($CT_HOSTNAME) an ..." >&2
  pct create "$CTID" "$TEMPLATE_STORAGE:vztmpl/$TEMPLATE" \
    --hostname "$CT_HOSTNAME" \
    --memory "$MEMORY" --swap 256 --cores "$CORES" \
    --rootfs "$STORAGE:$DISK" \
    --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
    --unprivileged 1 --features nesting=1 \
    --onboot 1 >&2
  # Tailscale needs the TUN device, which an LXC does not get by default.
  cat >> "/etc/pve/lxc/$CTID.conf" <<'CONF'
lxc.cgroup2.devices.allow: c 10:200 rwm
lxc.mount.entry: /dev/net/tun dev/net/tun none bind,create=file
CONF
fi

[ "$(pct status "$CTID" | awk '{ print $2 }')" = running ] || pct start "$CTID"
for _ in $(seq 1 60); do
  in_ct "getent hosts deb.debian.org >/dev/null" 2>/dev/null && break
  sleep 2
done

echo "Installiere Python, ssh und Tailscale ..." >&2
in_ct "export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq ca-certificates curl python3 openssh-client >/dev/null
  command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
  systemctl enable --now tailscaled >/dev/null
  id imgomat >/dev/null 2>&1 || useradd --system --home-dir /var/lib/img-o-mat --create-home --shell /usr/sbin/nologin imgomat" >&2

if ! in_ct "tailscale status >/dev/null 2>&1"; then
  echo "Melde den Container als $CT_HOSTNAME bei Tailscale an ..." >&2
  if [ -n "$TS_AUTHKEY" ]; then
    in_ct "tailscale up --hostname='$CT_HOSTNAME' --authkey='$TS_AUTHKEY'" >&2
  else
    echo "Den folgenden Link öffnen und den Container anmelden." >&2
    in_ct "tailscale up --hostname='$CT_HOSTNAME'" >&2
  fi
fi

echo "Kopiere den Dienst nach /opt/img-o-mat ..." >&2
ARCHIVE="$(mktemp)"
tar -C "$HERE" -czf "$ARCHIVE" service worker
pct push "$CTID" "$ARCHIVE" /tmp/img-o-mat.tgz
rm -f "$ARCHIVE"
in_ct "rm -rf /opt/img-o-mat.new && mkdir -p /opt/img-o-mat.new && tar -C /opt/img-o-mat.new -xzf /tmp/img-o-mat.tgz && rm /tmp/img-o-mat.tgz
  rm -rf /opt/img-o-mat && mv /opt/img-o-mat.new /opt/img-o-mat"

in_ct "set -e
  mkdir -p /etc/img-o-mat
  if [ ! -f /etc/img-o-mat/config.json ]; then
    python3 -c 'import json, sys; json.dump({\"root\": \"/var/lib/img-o-mat\", \"tokenFile\": \"/var/lib/img-o-mat/huggingface.token\", \"worker\": {\"ssh\": sys.argv[1]}}, open(\"/etc/img-o-mat/config.json\", \"w\"), indent=2)' '$WORKER'
  fi
  if [ ! -f /var/lib/img-o-mat/.ssh/id_ed25519 ]; then
    install -d -o imgomat -g imgomat -m 700 /var/lib/img-o-mat/.ssh
    su -s /bin/sh imgomat -c \"ssh-keygen -q -t ed25519 -N '' -C img-o-mat-service -f /var/lib/img-o-mat/.ssh/id_ed25519\"
  fi
  cat > /etc/systemd/system/img-o-mat.service <<'UNIT'
[Unit]
Description=img-o-mat (Bildmodelle für alle Apps)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
User=imgomat
Environment=IMG_O_MAT_CONFIG=/etc/img-o-mat/config.json
ExecStart=/usr/bin/python3 /opt/img-o-mat/service/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable img-o-mat >/dev/null 2>&1
  systemctl restart img-o-mat" >&2

NAME="$(in_ct "tailscale status --json 2>/dev/null | sed -n 's/.*\"DNSName\": *\"\\([^\"]*\\)\\.\".*/\\1/p' | head -n 1" || true)"
echo "Fertig: Container $CTID, im Tailnet als ${NAME:-$CT_HOSTNAME}, Seite http://${NAME:-$CT_HOSTNAME}:1236/" >&2
echo "CTID=$CTID"
in_ct "cat /var/lib/img-o-mat/.ssh/id_ed25519.pub"
