"""gmlx's additions to the Jev decision API: ``think: "auto"``, its request
settings and the server's request defaults."""

from __future__ import annotations

import pytest
from test_systemone_decide import DECISION_KEYS, THOUGHT, FakeEngine, _qs
from test_systemone_decide import _run as _run_decision

from gmlx.systemone import extensions
from gmlx.systemone.extensions import GMLX_EXTENSIONS, ignored_fields, request_schema
from gmlx.systemone.schema import SchemaError, jev_schema

NOUL = {"questions": {"q": {"type": "noul"}}}


def _run(body, **kw):
    return _run_decision(body, parse=request_schema,
                         decide=lambda schema, **k: extensions.decide(schema, "s", **k),
                         **kw)


# request_schema


def test_gmlx_extensions_list():
    assert GMLX_EXTENSIONS == ("think_threshold", "think_budget")


def test_think_auto_starts_without_a_thought_and_keeps_its_settings():
    s = request_schema({**NOUL, "think": "auto"})
    assert s["think"] == 0
    assert s["think_auto"] == {"threshold": 0.8, "budget": 64}
    s = request_schema({**NOUL, "think": "auto", "think_threshold": "0.9",
                        "think_budget": 128})
    assert s["think_auto"] == {"threshold": 0.9, "budget": 128}


def test_a_number_think_has_no_auto_settings():
    s = request_schema({**NOUL, "think": 32, "think_threshold": 0.5})
    assert (s["think"], s["think_auto"]) == (32, None)


def test_request_schema_matches_jev_schema_without_auto():
    body = {**NOUL, "think": 16, "samples": 2}
    assert request_schema(body) == {**jev_schema(body), "think_auto": None}


def test_jev_schema_refuses_think_auto():
    with pytest.raises(SchemaError, match="think must be a thought budget"):
        jev_schema({**NOUL, "think": "auto"})


@pytest.mark.parametrize("extra,match", [
    ({"think_threshold": 0}, "think_threshold must be above 0"),
    ({"think_threshold": 1.5}, "think_threshold must be above 0"),
    ({"think_threshold": "high"}, "think_threshold must be a number"),
    ({"think_budget": 0}, "think_budget must be 1 to 4096"),
    ({"think_budget": 4097}, "think_budget must be 1 to 4096"),
    ({"think_budget": "long"}, "think_budget must be an integer"),
])
def test_bad_think_auto_settings_are_refused(extra, match):
    with pytest.raises(SchemaError, match=match):
        request_schema({**NOUL, "think": "auto", **extra})


def test_a_bad_think_is_refused_with_the_auto_choice_named():
    with pytest.raises(SchemaError, match='0 to 4096, or "auto"'):
        request_schema({**NOUL, "think": "long"})


def test_request_schema_defaults_fill_only_absent_fields():
    auto = {"think": "auto", "think_threshold": 0.7, "think_budget": 32}
    s = request_schema(NOUL, defaults=auto)
    assert (s["think"], s["think_auto"]) == (0, {"threshold": 0.7, "budget": 32})
    s = request_schema({**NOUL, "think": 0}, defaults=auto)
    assert (s["think"], s["think_auto"]) == (0, None)
    s = request_schema({**NOUL, "think_budget": 16}, defaults=auto)
    assert s["think_auto"] == {"threshold": 0.7, "budget": 16}


def test_think_settings_are_ignored_without_auto():
    body = {**NOUL, "think": 8, "think_budget": 16}
    assert ignored_fields(body, request_schema(body)) == {"think_budget"}
    body = {**NOUL, "think": "auto", "think_budget": 16}
    assert ignored_fields(body, request_schema(body)) == set()


# decide with think: "auto"


class ThoughtEngine(FakeEngine):
    """Unsure without a thought in the prompt, sure with one."""

    def read(self, prompt, req):
        sure = THOUGHT[0] in prompt.ids
        self.weights = (lambda seed, i, n: [50.0] + [1.0] * (n - 1)) if sure \
            else (lambda seed, i, n: [1.0, 1.2] + [1.0] * (n - 2))
        return super().read(prompt, req)


THINK_AUTO_KEYS = {"threshold", "budget", "thought", "unsure", "confidence", "reads",
                   "total_ms"}


def test_think_auto_keeps_a_sure_first_pass():
    r = _run({**_qs("q1"), "think": "auto"})
    d = r["out"]["diagnostics"]
    auto = d["think_auto"]
    assert r["engine"].thinks == []
    assert set(auto) == THINK_AUTO_KEYS
    assert (auto["threshold"], auto["budget"], auto["thought"], auto["unsure"]) \
        == (0.8, 64, False, [])
    assert auto["confidence"] == {"q1": r["out"]["answers"]["q1"]["confidence"]}
    assert auto["reads"] == d["timing"]["reads"]
    assert set(d) == DECISION_KEYS | {"think_auto"}


def test_think_auto_runs_again_with_a_thought_when_an_answer_is_unsure():
    eng = ThoughtEngine()
    r = _run({**_qs("q1", "q2"), "think": "auto", "think_budget": 48}, engine=eng)
    out, d = r["out"], r["out"]["diagnostics"]
    auto = d["think_auto"]
    assert [t["budget"] for t in eng.thinks] == [48]
    assert (auto["budget"], auto["thought"], auto["unsure"]) == (48, True, ["q1", "q2"])
    assert all(c < 0.8 for c in auto["confidence"].values())
    assert out["answers"]["q1"]["label"] == "yes"
    assert out["answers"]["q1"]["confidence"] > 0.9
    assert d["thought"]["tokens"] == len(THOUGHT)
    # Four unsure reads, then one sure read on the thought. Timing covers
    # both runs, and think_auto holds the first.
    assert auto["reads"] == 4
    assert d["timing"]["reads"] == 5
    # The rows are the returned run's, as for the same request with think 48.
    second = _run({**_qs("q1", "q2"), "think": 48}, engine=ThoughtEngine())
    assert r["rows"] == second["rows"]


def test_think_auto_lists_unsure_questions_in_question_order():
    r = _run({**_qs("zeta", "alpha"), "think": "auto"}, engine=ThoughtEngine())
    assert r["out"]["diagnostics"]["think_auto"]["unsure"] == ["zeta", "alpha"]


def test_think_auto_threshold_decides_what_is_unsure():
    r = _run({**_qs("q1"), "think": "auto", "think_threshold": 0.5},
             engine=ThoughtEngine())
    assert r["out"]["diagnostics"]["think_auto"]["thought"] is False
    assert r["engine"].thinks == []


def test_think_auto_ignores_skipped_questions():
    body = {**_qs("q1", "q2", q2={"ask_if": {"q1": ["no"]}}), "think": "auto"}
    r = _run(body)
    assert r["out"]["answers"]["q2"] is None
    assert r["out"]["diagnostics"]["think_auto"]["unsure"] == []
