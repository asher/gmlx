"""Server response_format on MTP models: the stock refusal is deferred, and
each row's grammar travels from the prefill to the speculative batch."""

import types

import pytest

from gmlx.serve.patches import spec_grammar as sg

pytest.importorskip("llguidance")


def _stock_like_generate(seen):
    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        if self.draft_model is not None and args.logits_processors is not None:
            raise ValueError("Structured response_format is not supported "
                             "with speculative decoding.")
        seen.append(args.logits_processors)
        return "ctx"
    return generate


def _rg_class(seen):
    cls = type("RG", (), {"generate": _stock_like_generate(seen),
                          "_make_logits_processors":
                          lambda self, args, input_ids=None:
                          list(args.logits_processors or ())})
    sg._install_defer(cls)
    sg._install_restore(cls)
    return cls


def _mtp():
    return types.SimpleNamespace(draft_model=object(), draft_kind="mtp")


def test_mtp_request_passes_the_stock_refusal_and_keeps_its_processors():
    seen = []
    cls = _rg_class(seen)
    procs = ["grammar"]
    args = types.SimpleNamespace(logits_processors=procs)
    assert cls.generate(_mtp(), "p", args=args) == "ctx"
    assert seen == [None]
    # Restored on return, for the routes that read it afterwards.
    assert args.logits_processors is procs
    assert not hasattr(args, sg._DEFERRED_ATTR)


def test_engine_thread_sees_the_processors_while_deferred():
    cls = _rg_class([])
    args = types.SimpleNamespace(logits_processors=None)
    setattr(args, sg._DEFERRED_ATTR, ["grammar"])
    assert cls._make_logits_processors(_mtp(), args) == ["grammar"]
    assert args.logits_processors == ["grammar"]


def test_non_mtp_drafter_keeps_the_stock_refusal():
    cls = _rg_class([])
    other = types.SimpleNamespace(draft_model=object(), draft_kind="eagle3")
    args = types.SimpleNamespace(logits_processors=["grammar"])
    with pytest.raises(ValueError, match="not supported"):
        cls.generate(other, "p", args=args)


def test_restored_when_generate_raises():
    def boom(self, prompt, images=None, audio=None, args=None, videos=None):
        raise RuntimeError("queue full")
    cls = type("RG", (), {"generate": boom})
    sg._install_defer(cls)
    args = types.SimpleNamespace(logits_processors=["grammar"])
    with pytest.raises(RuntimeError):
        cls.generate(_mtp(), "p", args=args)
    assert args.logits_processors == ["grammar"]


class _Spec:
    def __init__(self, draft_kind="mtp"):
        self.draft_kind = draft_kind


def _ppb_class(result):
    def generate(self, *a, **kw):
        self.logits_processors = []      # the stock prefill clears them
        return result
    cls = type("PPB", (), {"generate": generate})
    sg._install_build(cls, _Spec)
    return cls


def _grammar_proc():
    import llguidance as llg
    from mlx_vlm.structured import LLGuidanceLogitsProcessor
    tok = types.SimpleNamespace(vocab_size=64)
    return LLGuidanceLogitsProcessor(llg.grammar_from("regex", "a"), tok)


def test_build_hangs_one_grammar_per_row(monkeypatch):
    built = []
    monkeypatch.setattr(
        "gmlx.spec.grammar.SpecGrammar.from_processors",
        classmethod(lambda cls, procs: built.append(procs) or "g"))
    result = _Spec()
    ppb = _ppb_class(result)()
    ppb.uids = [1, 2]
    ppb.logits_processors = [[], ["grammar"]]
    assert ppb.generate() is result
    assert result._kq_grammars == [None, "g"]
    assert built == [["grammar"]]


def test_build_skips_rows_without_processors_and_plain_batches():
    for result, procs in ((_Spec(), [[], []]), (object(), [["grammar"]]),
                          (_Spec("eagle3"), [["grammar"]])):
        ppb = _ppb_class(result)()
        ppb.uids = list(range(len(procs)))
        ppb.logits_processors = procs
        assert not hasattr(ppb.generate(), "_kq_grammars")


def test_start_stashes_grammars_on_the_first_cache_entry():
    starts = []
    cls = type("Spec", (), {"_start_rounds":
                            lambda self: starts.append(self._rounds_iter)})
    sg._install_start(cls)
    batch = cls()
    batch.draft_kind = "mtp"
    batch._rounds_iter = None
    batch.prompt_cache = [types.SimpleNamespace()]
    batch._kq_grammars = ["g"]
    batch._start_rounds()
    assert batch.prompt_cache[0]._kq_spec_grammars == ["g"]
    assert starts == [None]
