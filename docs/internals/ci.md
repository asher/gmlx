# CI and the GPU gate

A pull request needs two kinds of check before it can merge. The CI test
legs run the suite on GitHub's macOS runners, which have no usable GPU, so
they run it on the CPU. The `gate/gpu` status comes from a run of the GPU
tests on a Mac, recorded in git and pushed with the branch.

- [Gate a branch](#gate-a-branch)
- [What the gate runs](#what-the-gate-runs)
- [The gate/gpu status](#the-gategpu-status)
- [How results reach GitHub](#how-results-reach-github)
- [What CI runs](#what-ci-runs)
- [Flaky tests](#flaky-tests)
- [Nightly tests](#nightly-tests)
- [Suite growth](#suite-growth)

## Gate a branch

Set up the clone once, then run the gate on the commit you push:

```sh
python scripts/gate.py setup    # once per clone
git push                        # CI starts
python scripts/gate.py run      # meanwhile, on the Mac
git push                        # publishes the result
```

`run` tests the committed HEAD, so commit first. It records a pass or a
failure for the commit's tree, and the second `git push` publishes it. When
CI ends, it reads the result and sets `gate/gpu`.

When the gate ends after CI, nothing reads the result yet. Push again, or
refresh the status:

```sh
gh workflow run gate.yml -f pr=<number>
```

Every code push needs a new gate run, including a rebase onto a newer
main. A push that changes only docs keeps the status of the tree it builds
on, so it needs no run.

## What the gate runs

The gate runs the tests whose results depend on the device, on the GPU,
with 8 workers:

- the numerics folders under `tests/`: `models`, `cache`, `image`, `gen`,
  `spec`, `distill`, `tune`, `load` and `stream`;
- every other test file that imports `mlx` or `mlx_kquant`, or runs code
  that does in a child interpreter.

CI runs everything else. To list the gate's tests:

```sh
python scripts/gate.py scope
```

When the change touches concurrency code, such as `gmlx/container`,
`gmlx/stream` or the server lifecycle, the gate also runs those tests
under load, three times each.

A test that reaches the GPU through gmlx code, without importing mlx
itself, falls outside the gate. Such a test still runs the same Python on
CI's CPU backend. The full run before each release covers it.

### The full run

```sh
python scripts/gate.py run --full
```

The full run takes every test, nightly ones included, on the GPU and then
on the CPU. It also does two jobs for later runs:

- It checks the gate's scope against the two runs. A test file whose cases
  skip on one device and run on the other, outside the gate's tests,
  fails the run. Add the file to `GPU_EXTRA` in `scripts/gate.py`.
- It writes the test map that `scripts/affected_tests.py` reads, from the
  coverage of the GPU run.

Run it before each release, and when `affected_tests.py` reports a test
map over four weeks old.

## The gate/gpu status

| Status | Meaning |
|--------|---------|
| success, `passed in 4m` | The newest run on this exact tree passed. |
| failure, `failed` | The newest run on this exact tree failed. Fix it, commit and run the gate again. |
| success, `inherited from <commit>, docs-only delta` | This tree differs only in docs from a tree that passed. |
| success, `no code change` | The pull request changes only docs. |
| pending, `no gate result for this tree` | No Mac has gated this tree. Run the gate and push. |

A result for the exact tree always wins over an inherited one. When the
tree has more than one result, from reruns or from other Macs, the newest
counts. Results expire after four weeks.

To see the status that HEAD would get, and whether your result has reached
GitHub:

```sh
python scripts/gate.py status
```

## How results reach GitHub

A result is a git ref under `refs/gmlx-gate/`, named for the tree it
tested. `gate.py setup` configures the clone so that a plain `git push`
sends the current branch and this Mac's results together. That setting
changes what a plain push does in this clone: it pushes the current branch
to the branch of the same name.

Other forms of push carry the branch only. That includes
`git push -u origin <branch>`, and a plain push fails from a detached
HEAD. To send the results on their own:

```sh
git push origin 'refs/gmlx-gate-mine/*:refs/gmlx-gate/*'
```

Results stay on the Mac that ran the gate. When you push from another
machine, push the results from the Mac, then refresh the status with
`gh workflow run`.

`python scripts/gate.py prune` deletes results older than four weeks, on
this Mac and on GitHub.

## What CI runs

| Event | Test legs |
|-------|-----------|
| Pull request that changes code | `test (3.14)` runs the whole suite, pyright and the wheel checks. `test (3.11)` runs it without the numerics folders. |
| Pull request that changes only docs | `test (3.14)` runs the docs tests. `test (3.11)` passes without running. |
| Pull request that changes the ComfyUI pack | Adds `comfyui-pack (pin)`. |
| Push to main | `test (3.14)`. |
| Daily | `test (3.12)` and `test (3.13)` with the nightly tests, and the stress run. |
| Weekly | The drift report and the four ComfyUI pack legs. |
| Release tag | Three ComfyUI pack legs. |

Docs are `docs/`, `overrides/`, `mkdocs.yml` and Markdown files outside
`gmlx/` and `tests/`. Everything else counts as code.
`scripts/changed_paths.py` holds this rule for CI and the gate.

## Flaky tests

When tests fail, CI reruns the failed ones alone. A test that passes on
the rerun is flaky: the leg passes with a warning, and CI opens an issue
labelled `flaky` for the test. The issue closes itself after 20 green runs
in a row on main, or after 5 daily runs that no longer have the test.

## Nightly tests

Mark a costly test that rarely breaks as `nightly`:

```python
@pytest.mark.nightly
def test_every_import_order_binds_the_owned_classes():
    ...
```

Pull requests, pushes to main and the regular gate skip it. The daily legs
and the full gate run it.

## Suite growth

The `test (3.14)` leg of a pull request compares its test count and test
time with the last run on main, in the job summary. A notice in the check
names a change of 100 tests or more, or 60 seconds or more of new tests.
