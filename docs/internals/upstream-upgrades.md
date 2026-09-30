# Upstream releases

gmlx carries mlx-vlm 0.7.3 and mlx-lm 0.31.3 whole, in
`gmlx/_vendor/mlx_vlm` and `gmlx/_vendor/mlx_lm`, and imports those copies,
never an installed mlx-vlm or mlx-lm. The copy is gmlx code. A new upstream
release changes nothing in gmlx until a port takes one of its changes, by
decision and with a test. mlx stays an installed dependency, held to a
version floor.

## The copy

Each copied file is upstream's, with only its module references rewritten
from `mlx_vlm` and `mlx_lm` to `gmlx._vendor.mlx_vlm` and
`gmlx._vendor.mlx_lm`. `scripts/vendor_upstream.py` made the copy and does
the rewrite. `gmlx/_vendor/manifest.json` lists every file that upstream
shipped, with its sha256 as shipped and a status:

| Status | Meaning |
|--------|---------|
| `verbatim` | The file is upstream's bytes. |
| `codemod` | The file is upstream's bytes after the rewrite. |
| `edited` | The file was changed by hand, with a `# gmlx:` comment at each change. |
| `alias` | The file is replaced by a module that aliases an owned gmlx module. |
| `excluded` | The file is left out of the tree. |
| `license` | The file is the package's LICENSE, as upstream shipped it. |

A bug in the copy is fixed in place, like any gmlx bug. The fix carries a
`# gmlx:` comment and a test that fails without it, and
`python scripts/vendor_upstream.py manifest` records the new file. Code
that gmlx owns outright replaces its copy. The KV cache classes live in
`gmlx/cache/kv_cache.py`, and the copied cache module is an alias of it.
An owned model has a row in `gmlx/models/owned.py`, which the copied
loaders read before their own model packages.

A line that must name a stock module, such as a reader of the module names
an earlier release recorded, keeps its names with a `# gmlx: stock-name`
comment. Only the files that `STOCK_NAME_FILES` in the tool lists may carry
it.

## Checks

| Check | Where | What it does |
|-------|-------|--------------|
| Copy check | `vendor_upstream.py rewrite --check` and `verify` in the lint job, `tests/test_vendor_manifest.py` | Every module reference names the copy, and the tree matches the manifest. |
| Stock imports | `tests/stock_blocker.py`, `tests/test_no_stock_imports.py` | The tests refuse a stock import, and a child with poisoned stock packages runs gmlx without reaching one. |
| Claims | `gmlx/upstream/claims.py` and `tests/upstream/test_claims.py` | Each copied name that an installer rebinds exists in the copy and is gmlx's after install. `python -m gmlx.upstream.claims --installed` runs it. |
| mlx floor | `check_mlx_version`, at CLI entry | It refuses to run on an mlx below the floor that the copied mlx-vlm declares. `gmlx doctor` prints the installed mlx and the copied versions. |
| Static calls | `pyright`, in the macOS CI job | It type-checks the files in `[tool.pyright].include`, including their calls into the copy. |

The copy check also fails a copied file whose sha256 differs from the
manifest, a file under the copy that the manifest does not list, and an
edit without its marker. The poisoned child runs the CLI, loads, generates
and serves.

## Reading a new release

`scripts/upstream_canary.sh` downloads the wheels of the newest mlx-vlm and
mlx-lm, or of the versions it is given, together with the copied releases,
and runs `scripts/drift_report.py` on them. Nothing is installed. The
weekly drift job in CI runs it and puts the report in the run summary, and
it never fails the run.

```sh
scripts/upstream_canary.sh                 # the newest releases
scripts/upstream_canary.sh 0.7.4          # an mlx-vlm version, the newest mlx-lm
```

The report starts with a file summary for each package: the files that
upstream changed, removed or added, and the files that both upstream and
gmlx changed. Unit rows follow for the files that gmlx edits or imports
from, and for owned code taken from an upstream file. A unit is a
function, a method, the rest of a class or a module assignment. Each row
says whether upstream changed the unit, gmlx did, or both did, from a
three-way compare of the copy, the copied release and the new release. The
import rewrite is applied to both releases first, so it never shows as a
change.

`scripts/drift_report.py` also runs on its own. `--new mlx_vlm=DIR` and
`--old mlx_vlm=DIR` take package directories, and `--json` writes the same
data as JSON.

## Porting a change

The report is a list to triage, not a gate. Each change goes through three
steps:

1. Decide. Port an upstream correctness fix when the bug applies to the
   copy. List performance and structural changes for review, and leave the
   rest.
2. Port. Edit the copied file in place, or the owned module that replaced
   it, with a `# gmlx:` comment at the change. Add a test that fails
   without the port.
3. Record. Run `python scripts/vendor_upstream.py manifest` and then
   `verify`, and commit the port with its test and the manifest.

A port never moves the copy to a new release wholesale. The manifest keeps
the sha256 of the release each file was copied from, so the next report
still compares against it, and a ported unit that matches the new release
drops out of the rows.

## Upgrading mlx

mlx is the one upstream package that gmlx installs. mlx-kquant pins the
exact mlx release that its kernels were built against, so `pyproject.toml`
gives mlx no version of its own. `MLX_FLOOR` in `gmlx/upstream/versions.py`
is the floor that the copied mlx-vlm declares, and
`tests/upstream/test_versions.py` holds the two equal. These steps move
mlx:

1. Take the mlx-kquant release built for the new mlx.
2. Run the full test suite and the tests marked `integration` in a scratch
   venv, as [Testing](testing.md) describes.
3. Run live smoke tests with models on disk. Cover a served MTP model, a
   warm prompt-cache hit, an MXFP4 model, a DeepSeek-V4 model and one
   `gmlx talk` turn.
4. Commit the version bump on its own.

## Adding a rebind

New behavior in the copy goes in as an in-place edit, marked `# gmlx:`,
which needs no installer. An installer that rebinds a name in the copy
gets a row in `CLAIMS` in `gmlx/upstream/claims.py` in the same change.
The row names the copied module and attribute, the gmlx site that uses it,
and how the name is claimed: as a wrap, an HTTP route, the app or a
subclass. The check fails when the name is missing from the copy, or when
the installers leave the copy's own object bound to it. An installer is
idempotent, and it raises rather than doing nothing when the name it
expects is missing.
