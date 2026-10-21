#!/bin/sh
# Smoke test for the built site (Phase 7, D-045): serve build/site under the same sub-path
# GitHub Pages uses, load it in headless Chrome, and check the first screen rendered:
# the title, the F1 chart and the tour's buttons. No extra dependencies: GitHub's Ubuntu
# runners ship Chrome, and Python serves the files.
set -eu
cd "$(dirname "$0")/.."
CHROME="${CHROME:-$(command -v google-chrome || command -v chromium || echo "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")}"
# A free port chosen by the OS, so a server already running elsewhere can't answer instead.
PORT="${PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
ROOT="$(mktemp -d)"
ln -s "$PWD/build/site" "$ROOT/fraud-graph-detection"
python3 -m http.server "$PORT" -b 127.0.0.1 -d "$ROOT" >/dev/null 2>&1 &
SERVER=$!
# Stop the server on exit, keeping the script's own exit status.
trap 'status=$?; kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; rm -rf "$ROOT"; exit $status' EXIT
for _ in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf "http://127.0.0.1:$PORT/fraud-graph-detection/" >/dev/null && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/fraud-graph-detection/" >/dev/null || { echo "FAIL: server did not start"; exit 1; }

check() {  # check <hash route> <text expected in the rendered page>...
  url="http://127.0.0.1:$PORT/fraud-graph-detection/#$1"; shift
  dom="$("$CHROME" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=5000 --dump-dom "$url" 2>/dev/null)"
  for want in "$@"; do
    printf '%s' "$dom" | grep -q -- "$want" || { echo "FAIL $url: missing '$want'"; exit 1; }
  done
  echo "ok   $url"
}

check "/" "Investigating Bitcoin transactions flagged as illicit" "Illicit F1 per time step" \
  "Open the top alert" "CC BY-NC-ND 4.0"
check "/alerts/40" "Alert queue" "Risk score"
check "/patterns/43" "Communities with the most flagged transactions"
