# Upgrading mlx-vlm, mlx-lm and mlx

gmlx is a patch layer over stock mlx-vlm and mlx-lm. It installs late-bound
patches over private upstream symbols, deep-imports model internals, and
leaves everything between those seams stock. The inventory is the `SEAMS`
table in `gmlx/upstream/seams.py`, well over a hundred entries, and
`python -m gmlx.upstream.seams` prints the current count with its drift
report. Every seam is fragile by design: upstream point releases move the
symbols, so the surface is safe only under the versions this page
qualifies. It is the maintainer's procedure for changing them.

The versions are declared in `pyproject.toml` in three different ways.
mlx-vlm is an exact pin, `mlx-vlm==X.Y.Z`, because it owns the seams.
mlx-lm and mlx-kquant carry floors, `mlx-lm>=0.31` and
`mlx-kquant>=0.4.7,<0.5`. mlx itself is unconstrained there and arrives
through mlx-kquant, which pins the exact mlx release its kernels were built
against, and CI installs that same mlx explicitly. Three checks keep an
environment inside those bounds:

| Layer | Where | What it does |
|-------|-------|--------------|
| declared versions | `pyproject.toml` | the exact mlx-vlm pin and the mlx-lm and mlx-kquant floors described above |
| seam contract | `tests/upstream/test_upstream_seams.py` | every patched symbol is pinned to a source fingerprint. Drift fails CI naming the seam |
| runtime gate | `check_upstream_versions`, at CLI entry | mlx, mlx-lm or mlx-vlm below its floor refuses to run with an upgrade message, and newer than the qualified set warns once. `gmlx doctor` is exempt |

## Watching upstream releases

When mlx-vlm or mlx-lm publishes a release:

```sh
scripts/upstream_canary.sh
```

This builds a disposable venv with this checkout plus the latest mlx-vlm
and runs the seam check. A pass means the release is likely a safe bump,
still to be qualified by the procedure below, while a failure lists each
changed symbol and the gmlx site that uses it.

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

Any new patch or deep import of upstream internals gets a row in `SEAMS`
in the same change, then a regen. A seam that correctness or a hard feature
dependency relies on sets `critical=True`, and its installer must raise
when the seam is missing, whereas optional accelerations warn once and fall
back. Either way a patch is idempotent and never silently no-ops when the
upstream surface it expects has changed.

KV-cache classes have two origins since mlx-vlm 0.6.4 vendored its own. The
rules for isinstance checks and construction are in the docstring of
`gmlx/cache/compat.py`.
