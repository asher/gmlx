# vhs tape for the README demo GIF.
#
# Record with:
#   brew install vhs
#   vhs docs/assets/demo.tape        # writes docs/assets/demo.gif
#
# Run it from a scratch directory and pre-pull the model once before recording,
# so the pull replay is the instant already-downloaded path and the whole tape
# stays under ~45 seconds:
#   gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
#
# The Sleep durations below are first guesses; tune them against the actual
# pull-replay and load/generate timings on the recording machine before
# publishing the GIF.

Output docs/assets/demo.gif

Set FontSize 16
Set Width 960
Set Height 540
Set TypingSpeed 40ms

Type "gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to ."
Sleep 500ms
Enter
Sleep 4s

Type `gmlx run Qwen3-0.6B-Q4_K_M.gguf --prompt "Explain KV caching in one sentence." --max-tokens 60`
Sleep 500ms
Enter
Sleep 25s

Sleep 3s
