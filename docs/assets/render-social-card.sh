#!/bin/bash
# Render docs/assets/social-card.png, the link preview image of the docs site
# and the repository, from social-card.html with headless Chrome.
#
#   docs/assets/render-social-card.sh
#
# The fonts load from Google Fonts, so the render needs a network. After a
# change, upload the new PNG as the repository's social preview too, under
# Settings, General.
set -eu
set -o pipefail

CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/social-card.png}
PROFILE=$(mktemp -d)

# Headless Chrome can stay up after it writes the file, so stop it then.
"$CHROME" --headless=new --hide-scrollbars --force-device-scale-factor=1 \
  --window-size=1280,640 --virtual-time-budget=5000 \
  --user-data-dir="$PROFILE" --screenshot="$OUT" \
  "file://$HERE/social-card.html" 2>"$PROFILE/chrome.log" &
pid=$!
written=0
for _ in $(seq 120); do
  if grep -q 'bytes written to file' "$PROFILE/chrome.log"; then
    written=1
    break
  fi
  sleep 0.5
done
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
if [ "$written" != 1 ]; then
  echo "Chrome wrote no screenshot; its log is $PROFILE/chrome.log" >&2
  exit 1
fi
echo "wrote $OUT"
