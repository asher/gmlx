"""Unit tests for the over-generation probe helpers (no model forward pass)."""
import json

from gmlx import overgen


class _FakeTok:
    """Minimal stand-in: a settable EOG set, a deterministic chat template
    whose generation prompt appends [60, 70], and a fixed encode()."""

    def __init__(self):
        self.eos_token_ids = {1, 2}

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        ids = [9, 50, 120]  # conversation-start scaffold + <|user|> + content
        return ids + [60, 70] if add_generation_prompt else list(ids)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False  # bridge body must not add specials
        return [200, 201]


def test_suppressed_eos_clears_then_restores():
    tok = _FakeTok()
    with overgen.suppressed_eos(tok) as real:
        assert real == {1, 2}
        assert tok.eos_token_ids == set()
    assert tok.eos_token_ids == {1, 2}


def test_suppressed_eos_restores_on_exception():
    tok = _FakeTok()
    try:
        with overgen.suppressed_eos(tok):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tok.eos_token_ids == {1, 2}


def test_assistant_open_tokens_is_generation_prompt_suffix():
    assert overgen.assistant_open_tokens(_FakeTok()) == [60, 70]


def test_build_critique_bridge_reuses_seam_then_text_then_opener():
    bridge = overgen.build_critique_bridge(_FakeTok(), 50, "any bugs?")
    # seam token + encode("\nany bugs?") + assistant-open suffix
    assert bridge == [50, 200, 201, 60, 70]


def test_build_critique_bridge_without_seam_token():
    bridge = overgen.build_critique_bridge(_FakeTok(), None, "x")
    assert bridge == [200, 201, 60, 70]


class _SpyTok(_FakeTok):
    """Records the kwargs each apply_chat_template call receives."""

    def __init__(self):
        super().__init__()
        self.template_calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        self.template_calls.append(kw)
        return super().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, **kw
        )


def test_bridge_forwards_template_kwargs_for_no_thinking():
    tok = _SpyTok()
    overgen.build_critique_bridge(tok, 50, "x", {"enable_thinking": False})
    assert tok.template_calls  # the assistant-open diff rendered the template
    assert all(c.get("enable_thinking") is False for c in tok.template_calls)


def test_collect_interim_eos_finds_replayed_stops():
    got = overgen.collect_interim_eos([5, 2, 7, 1, 2], {1, 2})
    assert got == [
        {"index": 1, "token_id": 2},
        {"index": 3, "token_id": 1},
        {"index": 4, "token_id": 2},
    ]


def test_collect_interim_eos_empty_when_none_match():
    assert overgen.collect_interim_eos([5, 7, 8], {1, 2}) == []


def test_seam_marker_includes_id_and_text():
    m = overgen.seam_marker({"token_id": 154827, "text": "<|user|>"})
    assert "id=154827" in m
    assert "'<|user|>'" in m


def test_seam_marker_handles_empty_text():
    m = overgen.seam_marker({"token_id": 7, "text": ""})
    assert "id=7" in m
    assert "''" not in m  # no empty-repr noise when text is blank


def test_append_log_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "sub" / "overgen.jsonl"
    overgen.append_log(str(path), {"a": 1})
    overgen.append_log(str(path), {"b": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]
