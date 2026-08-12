# vhs tape for the README demo GIF.
#
# Record with:
#   brew install vhs
#   vhs docs/assets/demo.tape        # writes docs/assets/demo.gif
#
# What it shows, in one scene: a 27B model opened by bare id, loading in
# seconds, streaming a markdown answer with the live tok/s ticker visible.
# That is the differentiated story; keep the whole GIF under ~20 seconds.
#
# Prerequisites on the recording machine:
#   - a server config written by gmlx init that defines the id below.
#     Any mid-size id works; edit the Type lines to match your config.
#   - the model file already local. Nothing downloads on camera.
#   - a non-thinking model or intent. On-camera reasoning is dead air;
#     if the reply starts with a think block, switch id or @intent.
#
# The Sleep durations are first guesses. Tune them against live timings
# before publishing: the load sleep should end just after the first tokens
# appear, and the reply sleep just after the answer completes.

Output docs/assets/demo.gif

Set FontSize 18
Set Width 960
Set Height 540
Set TypingSpeed 40ms

Type "gmlx chat qwen3.6-27b-q6"
Sleep 500ms
Enter
# Zero-copy load: a 27B Q6_K reaches the prompt in a few seconds.
Sleep 4s

Type "Show a Python one-liner that flattens a nested list, then explain it in two sentences."
Sleep 500ms
Enter
# Streaming markdown with the live tok/s ticker; a compact ~200-token reply.
Sleep 8s

Type "/exit"
Sleep 500ms
Enter
Sleep 2s
