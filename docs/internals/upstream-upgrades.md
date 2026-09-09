# Upgrading mlx-vlm, mlx-lm and mlx

gmlx patches about thirty private symbols across mlx-vlm and mlx-lm and
deep-imports model internals. The inventory is in `gmlx/upstream/seams.py`.
That surface is safe only under the exact mlx-vlm pin in `pyproject.toml`,
because upstream point releases change it. This page is the maintainer's
procedure for changing the pin.

Three checks enforce the pin:

| Layer | Where | What it does |
|-------|-------|--------------|
| exact pin | `pyproject.toml` | `mlx-vlm==X.Y.Z`. mlx and mlx-lm are pinned together, with a minimum version so source installs resolve |
| seam contract | `tests/upstream/test_upstream_seams.py` | every patched symbol is pinned to a source fingerprint. Drift fails CI naming the seam |
| runtime gate | `check_upstream_versions`, at CLI entry | versions below the minimum refuse to run with an upgrade message, newer than the qualified set warns. `gmlx doctor` is exempt |

## Watching upstream releases

When mlx-vlm or mlx-lm publishes a release:

```sh
scripts/upstream_canary.sh
```

This builds a disposable venv with this checkout plus the latest mlx-vlm and
runs the seam check. A pass means the release is likely a safe bump, still
to be qualified by the procedure below. On failure it lists each changed
symbol and the gmlx site that uses it.

## Bump procedure

1. Build a scratch venv with the target versions, never the dev venv:

   ```sh
   python3 -m venv /tmp/gmlx-bump && . /tmp/gmlx-bump/bin/activate
   pip install -e /path/to/gmlx && pip install mlx-vlm==<target>
   python -m gmlx.upstream.seams   # drift report
   ```

2. Re-audit each drifted seam. Diff the upstream source between the pinned
   and target versions. `pip download --no-deps` fetches both, then unzip
   and `diff -r` them. The seam entry's `used_by` names the gmlx site to
   re-verify. Adjust patches on a branch as needed.

3. Regenerate fingerprints in a fresh interpreter of the scratch venv, since
   regen refuses to run once the installers have patched the process:

   ```sh
   python -m gmlx.upstream.seams --regen
   ```

   This also records the qualified versions the runtime gate warns against.

4. Run the full test suite in the scratch venv.

5. Live smoke tests with models on disk. Serve Qwen3.6-35B-A3B with MTP for
   prefill and decode, serve gemma-4-12B dense with a warm prompt-cache hit,
   run gpt-oss-20b MXFP4, serve deepseek-v4 and take one `gmlx talk` turn.

6. The env-gated integration tests:

   ```sh
   KQUANT_TEST_MTP_GGUF=<path> pytest tests/spec/test_full_prompt_prefill.py \
       tests/models/test_qwen35_verify_fold.py
   ```

7. Commit the pin bump and the regenerated `gmlx/upstream/seams.json`
   together, as their own commit.

## Adding a new seam

Any new patch or deep import of upstream internals gets a row in `SEAMS` in
the same change, then a regen. A seam that correctness or a hard feature
dependency relies on sets `critical=True`. Its installer must raise when the
seam is missing. Optional accelerations warn once and fall back.

KV-cache classes have two origins since mlx-vlm 0.6.4 vendored its own. The
rules for isinstance checks and construction are in the docstring of
`gmlx/cache/compat.py`.
