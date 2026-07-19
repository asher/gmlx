# Upgrading mlx-vlm / mlx-lm / mlx

gmlx monkeypatches ~30 private symbols across mlx-vlm and mlx-lm and
deep-imports model internals (the full inventory lives in
`gmlx/upstream_seams.py`). That surface is safe **only under the exact
`mlx-vlm==` pin in pyproject** - upstream point releases move it. Example:
mlx-vlm 0.6.4 vendored mlx-lm's `switch_layers` module, which silently
un-matched our MoE expert leaf swap and surfaced as an opaque `gather_mm`
shape error at prefill.

Three layers of defense:

- **Exact pin** (`pyproject.toml`): `mlx-vlm==X.Y.Z`; mlx and mlx-lm move in
  lockstep with it (mlx-lm carries a floor so source/git installs still
  resolve).
- **Seam contract** (`tests/test_upstream_seams.py`): every patched symbol is
  pinned to a source fingerprint; any drift fails CI naming the exact seam.
- **Runtime gate** (`check_upstream_versions`, called at CLI entry): a stale
  venv resolves pip constraints only once, so below-floor versions refuse to
  run with an upgrade message, and versions newer than the qualified set warn.
  `gmlx doctor` is exempt so it can diagnose a broken env.

## Watching upstream releases

When mlx-vlm (or mlx-lm) cuts a release:

```sh
scripts/upstream_canary.sh
```

This builds a disposable venv with this checkout + the latest mlx-vlm and runs
the seam check. `canary PASS` means the release is likely a safe bump (still
qualify it below); a failure lists each moved/rewritten symbol and the
gmlx site that consumes it.

## Bump procedure

1. **Scratch venv** with the target versions (never the dev venv):

   ```sh
   python3 -m venv /tmp/gmlx-bump && . /tmp/gmlx-bump/bin/activate
   pip install -e /path/to/gmlx && pip install mlx-vlm==<target>
   python -m gmlx.upstream_seams   # drift report
   ```

2. **Re-audit each drifted seam.** Diff the upstream source between the pinned
   and target versions (`pip download --no-deps -d /tmp/old mlx-vlm==<pinned>`
   etc., unzip, `diff -r`). The seam entry's `used_by` names the gmlx site
   to re-verify; adjust patches on a branch as needed.

3. **Regenerate fingerprints** in a *fresh* interpreter of the scratch venv
   (regen refuses to run once our installers have patched the process):

   ```sh
   python -m gmlx.upstream_seams --regen
   ```

   This also records the qualified versions the runtime gate warns against.

4. **Full test suite** in the scratch venv: `pytest`.

5. **Live smokes** (models on disk):
   - Qwen3.6-35B-A3B MTP serve: prefill + decode (MoE + switch_layers + the
     whole spec-engine seam set);
   - gemma-4-12B dense serve + APC warm hit;
   - gpt-oss-20b MXFP4 (native-fp fused-GLU path);
   - a deepseek-v4 serve (vendored model path, expected unaffected);
   - one `gmlx talk` turn (STT/TTS routes).

6. **Env-gated integration tests**:

   ```sh
   KQUANT_TEST_MTP_GGUF=<path> pytest tests/test_full_prompt_prefill.py \
       tests/test_qwen35_verify_fold.py
   ```

7. **Pin bump commit**: pyproject pin(s) + the regenerated
   `gmlx/upstream_seams.json` together, as their own commit.

## Cache-class origins (`gmlx/cache_compat.py`)

mlx-vlm <= 0.6.3 re-exported mlx-lm's KV-cache classes; 0.6.4 vendored its
own `models/cache.py`, so the same cache kind has two class identities and
which one a live cache carries depends on who built it. Rules for gmlx
code:

- **isinstance checks** on upstream cache kinds go through
  `cache_compat.cache_types(name)` (every loaded origin), never a direct
  single-origin import.
- **Constructing** a cache that upstream machinery will consume uses the
  consumer's origin: `runtime_cache_module()` when the consumer is the
  mlx-vlm stack (apc, ar batches), `construction_cache_module()` in code
  shared between the vlm and pure-mlx-lm stacks (resolves to vlm when
  loaded, mlx-lm otherwise, without importing mlx-vlm), and a plain mlx-lm
  import only in mlx-lm-only decode paths (`stream_generate` CLI).
- Test fixtures standing in for runtime-built caches construct from
  `runtime_cache_module()` so they exercise the classes the serve stack
  actually produces on the pinned version.

## Adding a new seam

Any new monkeypatch or deep import of upstream internals gets a row in
`SEAMS` (`gmlx/upstream_seams.py`) in the same change, then
`python -m gmlx.upstream_seams --regen`. Critical seams (correctness or
a hard feature dependency) set `critical=True` and their installer must raise
when the seam is missing; optional accelerations warn once and fall back.
