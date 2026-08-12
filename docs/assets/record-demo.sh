#!/bin/bash
# Record the README demo GIF: one chat turn against a running gmlx server.
#
#   brew install asciinema agg tmux
#   gmlx serve                          # leave it up; the model stays resident
#   docs/assets/record-demo.sh          # writes docs/assets/demo.gif
#
# The demo shows the normal steady state: a server is already up, so `gmlx
# chat` connects to it as a plain client and the prompt appears at once. Warm
# the model with one request before recording, otherwise the first turn pays
# the load and the timings misrepresent a working setup.
#
# Timings are not scripted. The driver watches the pane and moves on when the
# prompt appears and when the turn's stats line lands, so the recording runs
# at the speed the machine actually delivers.
set -eu
set -o pipefail

MODEL=${MODEL:-qwen3.6-27b-q6-k@instruct}
QUESTION=${QUESTION:-"Write a Python LRU cache decorator, then explain it in two bullets."}
OUT=${OUT:-docs/assets/demo.gif}
COLS=${COLS:-110}
ROWS=${ROWS:-28}
HOLD=${HOLD:-2.2}                  # keep recording this long past the reply
LOOP_PAUSE=${LOOP_PAUSE:-5.0}      # still frame at the end, before the loop
FONT_SIZE=${FONT_SIZE:-16}
CAST_OUT=${CAST_OUT:-}             # set to keep the cast for re-rendering

# A thinking model spends the whole recording reasoning before it answers, so
# pick a non-thinking id or intent (@instruct on the Qwen families).

TM=$(command -v tmux)
SOCK=gmlx-demo
SESSION=d
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; $TM -L "$SOCK" kill-server 2>/dev/null || true' EXIT

pane() { $TM -L "$SOCK" capture-pane -p -t "$SESSION" 2>/dev/null; }

# Type one character at a time so the recording reads as a person typing.
type_str() {
  local s=$1 i
  for (( i = 0; i < ${#s}; i++ )); do
    $TM -L "$SOCK" send-keys -t "$SESSION" -l -- "${s:i:1}"
    sleep 0.035
  done
}

wait_for() {                            # wait_for REGEX TIMEOUT LABEL
  local re=$1 timeout=$2 label=$3 ticks=0 limit
  limit=$(( timeout * 2 ))              # half-second ticks
  while [ "$ticks" -lt "$limit" ]; do
    if pane | grep -qE -- "$re"; then
      echo "[rec] $label after $(( ticks / 2 ))s"
      return 0
    fi
    sleep 0.5
    ticks=$(( ticks + 1 ))
  done
  echo "[rec] gave up waiting for $label after ${timeout}s" >&2
  return 1
}

$TM -L "$SOCK" kill-server 2>/dev/null || true   # no server yet is the norm
# A throwaway cache keeps the operator's own prompt history out of the frame:
# prompt_toolkit would otherwise autosuggest past prompts as grey ghost text.
$TM -L "$SOCK" new-session -d -s "$SESSION" -x "$COLS" -y "$ROWS" \
  "env PS1='$ ' XDG_CACHE_HOME=$WORK/cache bash --norc --noprofile -i"
$TM -L "$SOCK" set -g status off
sleep 0.5

(
  # However this driver ends, close the session so the recording stops with
  # it instead of running to the asciinema timeout.
  trap '$TM -L "$SOCK" kill-session -t "$SESSION" 2>/dev/null || true' EXIT
  sleep 1.5
  type_str "gmlx chat $MODEL"
  sleep 0.4
  $TM -L "$SOCK" send-keys -t "$SESSION" Enter
  wait_for '^>>' 180 "chat ready"
  sleep 1.0
  type_str "$QUESTION"
  sleep 0.4
  $TM -L "$SOCK" send-keys -t "$SESSION" Enter
  wait_for '\[chat\] prompt' 240 "reply complete"
  sleep "$HOLD"
) &

# --window-size pins the geometry: a headless take otherwise falls back to
# 80x24 and rewraps every line.
asciinema rec --overwrite --window-size "${COLS}x${ROWS}" \
  --command "$TM -L $SOCK attach -t $SESSION" "$WORK/demo.cast"
wait

# End on the stats line. What follows is the session teardown, whose
# "[server exited]" notice reads as a crashed server in a looping GIF.
python3 - "$WORK/demo.cast" "$WORK/trim.cast" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
header, events = lines[0], lines[1:]
cut = max((i for i, ln in enumerate(events)
           if "[chat] prompt" in json.loads(ln)[2]), default=None)
if cut is None:
    sys.exit("no stats line in the cast: did the turn finish?")
kept = [json.loads(ln) for ln in events[:cut + 6]]
with open(dst, "w") as fh:
    fh.write(header + "\n")
    for ev in kept:
        fh.write(json.dumps(ev) + "\n")
PY

# A black background matching the TUI, and an idle limit high enough that agg
# never silently compresses a real pause into a shorter one.
# --last-frame-duration is what actually sets the rest before the loop
# restarts; agg otherwise caps that final frame at 3 seconds.
agg --font-size "$FONT_SIZE" --idle-time-limit 3600 \
  --last-frame-duration "$LOOP_PAUSE" \
  --theme "000000,e5e5e5,000000,cd3131,0dbc79,e5e510,2472c8,bc3fbc,11a8cd,e5e5e5,666666,f14c4c,23d18b,f5f543,3b8eea,d670d6,29b8db,ffffff" \
  "$WORK/trim.cast" "$OUT"
echo "[rec] wrote $OUT"

# The cast is the re-renderable source: keeping it means a later change of
# font size, theme, or end pause costs an agg run instead of a new recording.
if [ -n "$CAST_OUT" ]; then
  cp "$WORK/trim.cast" "$CAST_OUT"
  echo "[rec] wrote $CAST_OUT"
fi
