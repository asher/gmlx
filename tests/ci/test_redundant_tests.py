"""scripts/redundant_tests.py on small coverage databases: a test whose
coverage another test in its file holds is listed, and cases that take
distinct branches are not."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from coverage import CoverageData

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("_redundant_tests",
                                                  _ROOT / "scripts" / "redundant_tests.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rt = _load()


def _coverage(tmp_path: Path, runs: dict[str, dict[str, list[tuple[int, int]]]]) -> Path:
    """A branch coverage database: each test context ran the given arcs."""
    db = tmp_path / ".coverage"
    data = CoverageData(basename=str(db))
    for context, arcs in runs.items():
        data.set_context(context)
        data.add_arcs({str(tmp_path / f): a for f, a in arcs.items()})
    data.write()
    return db


_IF = [(-1, 1), (1, 2), (2, -1)]
_ELSE = [(-1, 1), (1, 3), (3, -1)]


def test_a_duplicate_case_and_a_test_inside_another_are_listed(tmp_path):
    db = _coverage(tmp_path, {
        "tests/test_a.py::test_sign[1]|run": {"gmlx/m.py": _IF},
        "tests/test_a.py::test_sign[2]|run": {"gmlx/m.py": _IF},
        "tests/test_a.py::test_both|run": {"gmlx/m.py": _IF + _ELSE},
        # Its setup is not its own coverage.
        "tests/test_a.py::test_sign[1]|setup": {"gmlx/m.py": _ELSE},
    })
    found = rt.candidates(rt.read_masks(db))
    assert found == {"tests/test_a.py": [
        ("tests/test_a.py::test_sign[2]", "same", "tests/test_a.py::test_sign[1]"),
        ("tests/test_a.py::test_sign[1]", "inside", "tests/test_a.py::test_both")]}


def test_cases_that_take_distinct_branches_are_not_listed(tmp_path):
    db = _coverage(tmp_path, {
        "tests/test_b.py::test_sign[1]|run": {"gmlx/m.py": _IF},
        "tests/test_b.py::test_sign[-1]|run": {"gmlx/m.py": _ELSE},
        # The same coverage in another file is not a duplicate.
        "tests/test_c.py::test_sign|run": {"gmlx/m.py": _IF},
    })
    masks = rt.read_masks(db)
    assert rt.candidates(masks) == {}
    assert rt.param_sets(masks, min_cases=2) == []


def test_cases_with_fewer_coverage_sets_than_cases_are_named(tmp_path):
    db = _coverage(tmp_path, {
        f"tests/test_d.py::test_sign[{n}]|run": {"gmlx/m.py": _IF if n % 2 else _ELSE}
        for n in range(6)})
    assert rt.param_sets(rt.read_masks(db)) == [("tests/test_d.py::test_sign", 6, 2)]


def test_a_test_with_no_coverage_is_left_out(tmp_path):
    db = _coverage(tmp_path, {
        "tests/test_e.py::test_child|run": {},
        "tests/test_e.py::test_here|run": {"gmlx/m.py": _IF},
    })
    assert rt.candidates(rt.read_masks(db)) == {}
