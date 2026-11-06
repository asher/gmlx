"""Synthetic alignment cases for Nemotron 3.5 Transcribe word timestamps:
a tiny SentencePiece model, transcripts with speaker tags, and CTC log
probabilities, 80 ms speaker probabilities and 10 ms diarization labels
that place each token where the case says.

Used by the oracle script ``ref_align.py``, which stores every input in
the fixture, so the CPU test reads the same arrays. Needs numpy and
sentencepiece.
"""

from __future__ import annotations

import io

import numpy as np

SEED = 29
SPEAKERS = 4
CORPUS_WORDS = (
    "the a an and of to in is was it for on with as at by from that this "
    "cat dog sat ran mat hat red blue green old new small big quick lazy "
    "river stone market garden window letter morning evening question "
    "answer number seven eleven twelve hundred thousand budget meeting "
    "schedule committee review quarter tuesday friday open closed").split()
CJK = "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u5427\u660e\u5929\u89c1"
EVENT = "<ev:laughter>"


def train_spm(vocab_size: int = 160) -> bytes:
    """A BPE model over the corpus words and the CJK characters, with the
    event tag as a user-defined piece (the real model's tags are)."""
    import sentencepiece as spm

    rng = np.random.default_rng(SEED)
    lines = [" ".join(rng.choice(CORPUS_WORDS, 12)) for _ in range(400)]
    lines += [CJK[i:i + 6] for i in range(0, len(CJK), 3)] * 20
    out = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines), model_writer=out, vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0, user_defined_symbols=[EVENT],
        bos_id=-1, eos_id=-1, unk_id=0, num_threads=1, minloglevel=2,
        )
    return out.getvalue()


def _words(rng, n: int) -> list[str]:
    return [str(w) for w in rng.choice(CORPUS_WORDS, n)]


def cases() -> dict[str, dict]:
    """Each case: ``turns`` as (speaker tag or None, diarization column or
    None, words, first CTC frame, last CTC frame); ``frames`` in 40 ms CTC
    frames; ``tags`` whether the transcript carries ``<spk:N>``;
    ``speakers`` whether speaker probabilities exist; ``columns`` the
    diarization columns. A turn's tokens spread over its frames."""
    rng = np.random.default_rng(SEED)
    out: dict[str, dict] = {}
    out["plain"] = dict(frames=150, tags=False, speakers=False, columns=SPEAKERS,
                        turns=[(None, None, _words(rng, 8) + [EVENT] + _words(rng, 4), 5, 140)])
    out["two"] = dict(frames=420, tags=True, speakers=True, columns=SPEAKERS, turns=[
        (0, 1, _words(rng, 10), 4, 120), (1, 3, _words(rng, 9), 130, 250),
        (0, 1, _words(rng, 6), 262, 330), (1, 3, _words(rng, 7), 340, 410)])
    long_turns = []
    start = 6
    for i in range(16):
        tag = i % 2
        n = int(rng.integers(8, 16))
        span = n * 9
        long_turns.append((tag, (2, 0)[tag], _words(rng, n), start, start + span))
        # Every fifth turn overlaps the next one by a second.
        start += span + (-25 if i % 5 == 4 else 12)
    out["long"] = dict(frames=(start + 21) // 2 * 2, tags=True, speakers=True, columns=SPEAKERS,
                       turns=long_turns)
    out["unmatched"] = dict(frames=360, tags=True, speakers=True, columns=2, turns=[
        (0, 0, _words(rng, 8), 4, 110), (1, 1, _words(rng, 8), 120, 230),
        (2, None, _words(rng, 6), 240, 350)])
    out["cjk"] = dict(frames=120, tags=False, speakers=True, columns=SPEAKERS,
                      turns=[(None, 0, [CJK[:4], CJK[4:9], CJK[9:13]], 4, 110)])
    out["too_long"] = dict(frames=12, tags=False, speakers=False, columns=SPEAKERS,
                           turns=[(None, None, _words(rng, 20), 0, 11)])
    return out


def transcript(case: dict) -> str:
    parts = []
    for tag, _, words, _, _ in case["turns"]:
        if case["tags"]:
            parts.append(f"<spk:{tag}>")
        parts.extend(words)
    return " ".join(parts)


def _is_cjk(word: str) -> bool:
    return all("\u4e00" <= c <= "\u9fff" for c in word)


def _char_ids(sp, c: str) -> list[int]:
    i = sp.piece_to_id(c)
    return [i] if i != sp.unk_id() else [t for t in sp.encode(c)
                                          if sp.id_to_piece(t) != "\u2581"]


def inputs(case: dict, sp, name: str) -> dict[str, np.ndarray]:
    """``logp`` (frames, vocab + 1; blank last; float16), ``speaker_probs`` (frames
    / 2, columns) and ``labels`` (columns, frames * 4), for one case."""
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    frames = case["frames"]
    vocab = sp.get_piece_size() + 1
    blank = vocab - 1
    logits = rng.normal(0.0, 0.7, (frames, vocab))
    logits[:, blank] += 3.0
    activity = np.full((frames, case["columns"]), 0.02)
    for tag, column, words, first, last in case["turns"]:
        ids = [t for w in words for t in (
            [i for c in w for i in _char_ids(sp, c)] if _is_cjk(w) else sp.encode(w))]
        span = max(1, last - first)
        for k, t in enumerate(ids):
            f = first + int(round((k + 0.5) * span / (len(ids) + 0.5)))
            logits[min(f, frames - 1), t] += 7.0 + rng.uniform(0, 1.5)
            if rng.uniform() < 0.3 and f + 1 < frames:
                logits[f + 1, t] += 5.0
        if column is not None:
            ramp = np.clip(np.minimum(np.arange(frames) - first + 3,
                                      last + 3 - np.arange(frames)) / 3.0, 0, 1)
            activity[:, column] = np.maximum(activity[:, column], 0.02 + 0.93 * ramp)
    logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) \
        - logits.max(1, keepdims=True)
    # float16 halves the fixture; both sides read it back as float32.
    out = {"logp": logp.astype(np.float16)}
    if case["speakers"]:
        probs = activity.reshape(frames // 2, 2, -1).mean(1)
        out["speaker_probs"] = np.clip(probs + rng.normal(0, 0.01, probs.shape), 0, 1
                                       ).astype(np.float32)
        fine = np.repeat(activity, 4, axis=0) + rng.normal(0, 0.03, (frames * 4, activity.shape[1]))
        out["labels"] = (fine > 0.4641).T.copy()
    return out
