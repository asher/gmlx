# Working on gmlx

Read these before you change code or tests:

- [Testing](docs/internals/testing.md): the test tiers, and how to write a
  test.
- [CI and the GPU gate](docs/internals/ci.md): what a pull request needs to
  merge.

While you work, run the tests your change can break:

```sh
python scripts/affected_tests.py --run
```

Before you push a code change, run the GPU gate on a Mac, then push:

```sh
git commit ...                # the gate refuses uncommitted changes
python scripts/gate.py run
git push                      # sends the branch and the gate result
```

Only a bare `git push` sends the result, and only in a clone where
`python scripts/gate.py setup` has run. `git push origin <branch>` sends
the branch alone.

A result belongs to the tree it tested, so any later code change needs a
new run. To see whether this tree has a result, and whether it has reached
GitHub:

```sh
python scripts/gate.py status
```
