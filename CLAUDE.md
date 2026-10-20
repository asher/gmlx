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

Before you push a code change, commit it and run the GPU gate on a Mac:

```sh
python scripts/gate.py run
```
