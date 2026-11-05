#!/bin/sh
# On oi-o-mat: rolls out every new main by itself. launchd starts one round every five minutes; a round fetches
# origin/main and, when it differs from what was rolled out last, waits until nothing paints or downloads, moves this
# checkout forward, runs the tests and then deploy/to-proxmox.sh (which also runs install-worker.sh, so worker changes
# and new runner pins arrive too).
#
#   deploy/autodeploy.sh               one round (what launchd runs)
#   deploy/autodeploy.sh --install     start it with every login, every five minutes
#   deploy/autodeploy.sh --uninstall   stop it
#   deploy/autodeploy.sh --status      what was rolled out last and the end of the log
#
# The checkout must be on main without local changes; otherwise the round only logs why it does nothing. A commit whose
# tests fail is not tried again; a failed deploy is (Proxmox may just have been off). Failures also show up as a macOS
# notification. Settings for to-proxmox.sh (PROXMOX, WORKER, LOKI_URL, ...) go into ~/.config/img-o-mat/autodeploy.env
# as KEY=value lines. Log: ~/Library/Logs/img-o-mat/autodeploy.log.
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.marcelbenders.img-o-mat.autodeploy"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
STATE="${AUTODEPLOY_STATE:-$HOME/Library/Application Support/img-o-mat/autodeploy}"
LOG="${AUTODEPLOY_LOG:-$HOME/Library/Logs/img-o-mat/autodeploy.log}"
SETTINGS="${AUTODEPLOY_ENV:-$HOME/.config/img-o-mat/autodeploy.env}"
SERVICE="${IMG_O_MAT_URL:-http://img-o-mat:1236}"
# The tests replace these two; on oi-o-mat they stay as they are.
DEPLOY="${AUTODEPLOY_DEPLOY:-$HERE/deploy/to-proxmox.sh}"
PYTHON="${AUTODEPLOY_PYTHON:-/usr/bin/python3}"

log() {
  mkdir -p "$(dirname "$LOG")"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"
}

notify() {
  command -v osascript >/dev/null 2>&1 || return 0
  osascript -e "display notification \"$1\" with title \"img-o-mat\"" >/dev/null 2>&1 || true
}

case "${1:-}" in
  --install)
    mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
    # launchd starts with a bare PATH: uv and git from Homebrew, scutil from /usr/sbin.
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/sh</string><string>$HERE/deploy/autodeploy.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLIST
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "Läuft alle fünf Minuten. Log: $LOG"
    exit 0 ;;
  --uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Automatisches Ausrollen ist aus."
    exit 0 ;;
  --status)
    echo "Zuletzt ausgerollt: $(cat "$STATE/deployed" 2>/dev/null || echo nichts)"
    [ -f "$STATE/failed-tests" ] && echo "Tests scheiterten an: $(cat "$STATE/failed-tests")"
    [ -f "$LOG" ] && tail -n 15 "$LOG"
    exit 0 ;;
  -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "Unbekannt: $1" >&2; exit 2 ;;
esac

mkdir -p "$STATE"
# One round at a time: a deploy can take longer than five minutes when install-worker.sh moves a runner.
if ! mkdir "$STATE/lock" 2>/dev/null; then
  pid="$(cat "$STATE/lock/pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
  rm -rf "$STATE/lock"
  mkdir "$STATE/lock"
fi
echo $$ > "$STATE/lock/pid"
trap 'rm -rf "$STATE/lock"' EXIT INT TERM

cd "$HERE"
[ "$(git symbolic-ref --short -q HEAD || true)" = main ] || { log "Nicht auf main, nichts zu tun."; exit 0; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { log "Lokale Änderungen im Checkout, nichts zu tun."; exit 0; }

git fetch --quiet origin main || { log "git fetch scheiterte."; exit 1; }
NEW="$(git rev-parse FETCH_HEAD)"
DEPLOYED="$(cat "$STATE/deployed" 2>/dev/null || true)"
[ "$NEW" != "$DEPLOYED" ] || exit 0
[ "$NEW" != "$(cat "$STATE/failed-tests" 2>/dev/null || true)" ] || exit 0
SHORT="$(git rev-parse --short "$NEW")"

# Restarting the service cancels what paints or downloads; the next round tries again. If the service does not answer,
# a deploy is what may bring it back.
if BUSY="$(curl -fsS --max-time 10 "$SERVICE/status" 2>/dev/null)"; then
  if ! printf '%s' "$BUSY" | "$PYTHON" -c '
import json, sys
s = json.load(sys.stdin)
busy = s["painter"]["running"] or s["painter"]["queue"]
busy = busy or any(d.get("state") in ("waiting", "running") for d in s.get("downloads", {}).values())
sys.exit(1 if busy else 0)'; then
    log "$SHORT wartet: der Dienst malt oder lädt gerade."
    exit 0
  fi
fi

git merge --quiet --ff-only "$NEW" || { log "$SHORT lässt sich nicht vorspulen (main weicht ab)."; notify "Autodeploy: main lässt sich nicht vorspulen"; exit 1; }

if [ "${AUTODEPLOY_TESTS:-1}" != 0 ]; then
  if ! OUT="$("$PYTHON" -m unittest discover tests 2>&1)"; then
    printf '%s\n' "$OUT" | tail -n 20 >> "$LOG"
    log "$SHORT nicht ausgerollt: Tests scheitern."
    echo "$NEW" > "$STATE/failed-tests"
    notify "Autodeploy: Tests scheitern an $SHORT"
    exit 1
  fi
fi

log "Rolle $SHORT aus: $(git log -1 --format=%s "$NEW")"
if (
  if [ -f "$SETTINGS" ]; then set -a; . "$SETTINGS"; set +a; fi
  "$DEPLOY"
) >> "$LOG" 2>&1; then
  echo "$NEW" > "$STATE/deployed"
  rm -f "$STATE/failed-tests" "$STATE/failed-deploy"
  log "$SHORT ist ausgerollt."
else
  log "$SHORT: Ausrollen scheiterte, nächste Runde versucht es wieder."
  # One notification per commit, not one every five minutes.
  if [ "$(cat "$STATE/failed-deploy" 2>/dev/null || true)" != "$NEW" ]; then
    echo "$NEW" > "$STATE/failed-deploy"
    notify "Autodeploy: Ausrollen von $SHORT scheiterte"
  fi
  exit 1
fi
