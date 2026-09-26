# Upgrading mlx-vlm, mlx-lm and mlx

gmlx is a patch layer over stock mlx-vlm and mlx-lm. It installs
late-bound patches over private upstream symbols, deep-imports model
internals, and leaves everything between those seams stock. Upstream point
releases move the symbols, so the patch layer is safe only under a
qualified set of versions.

## How the versions are held

The `SEAMS` table in `gmlx/upstream/seams.py` lists every seam.
`python -m gmlx.upstream.seams` prints the seam count when every seam
matches the installed upstream, and the drift report otherwise.

`pyproject.toml` pins mlx-vlm exactly, because mlx-vlm owns most seams.
mlx-lm and mlx-kquant carry floors. mlx has no version there, because
mlx-kquant pins the exact mlx release that its kernels were built against,
and CI installs that same mlx. Three checks keep an environment inside
those bounds:

| Check | Where | What it does |
|-------|-------|--------------|
| Seam contract | `tests/upstream/test_upstream_seams.py` | It pins every patched symbol to a source fingerprint. Drift fails CI and names the seam. |
| Runtime gate | `check_upstream_versions`, at CLI entry | It refuses to run below a floor and warns once above the qualified set. `gmlx doctor` is exempt. |
| Static imports | `pyright`, in the macOS CI job | It checks the files in `[tool.pyright].include` against the installed upstream source. |

## Watching upstream releases

When mlx-vlm publishes a release, run `scripts/upstream_canary.sh`. It
builds a disposable venv with this checkout, lifts only the mlx-vlm pin to
the newest release and runs the seam check. A pass means the release is
likely a safe bump, still to be qualified by the
[bump procedure](#bump-procedure). A failure lists each changed symbol and
the gmlx site that uses it.

## Bump procedure

1. Build a scratch venv with the target versions, never the dev venv.

   ```sh
   python3 -m venv /tmp/gmlx-bump && . /tmp/gmlx-bump/bin/activate
   pip install -e /path/to/gmlx && pip install mlx-vlm==<target>
   python -m gmlx.upstream.seams   # drift report
   ```

2. Audit each drifted seam again. Diff the upstream source between the
   pinned and target versions, which `pip download --no-deps` fetches. The
   seam's `used_by` field names the gmlx site to verify again.

3. Regenerate the fingerprints in a fresh interpreter of the scratch venv,
   since the regeneration refuses to run once the installers have patched
   the process. It also records the qualified versions for the runtime
   gate.

   ```sh
   python -m gmlx.upstream.seams --regen
   ```

4. Run the full test suite and the tests marked `integration` in the
   scratch venv, as [Testing](testing.md) describes.

5. Run live smoke tests with models on disk. Cover a served MTP model, a
   warm prompt-cache hit, an MXFP4 model, a DeepSeek-V4 model and one
   `gmlx talk` turn.

6. Commit the pin bump and the regenerated `gmlx/upstream/seams.json`
   together, as their own commit.

## Adding a new seam

Any new patch or deep import of upstream internals gets a row in `SEAMS`
in the same change, then a regeneration. A seam that correctness or a hard
feature needs sets `critical=True`, and its installer raises when the
seam is missing. An optional acceleration warns once and falls back.
Either way, a patch is idempotent and never silently does nothing when the
upstream surface it expects has changed.

KV-cache classes come from both mlx-lm and mlx-vlm. The rules for
isinstance checks and construction are in the docstring of
`gmlx/cache/compat.py`.
