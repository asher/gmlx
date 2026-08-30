"""CLI kv-bits on a long prompt: prefill through a quantized cache.

Integration (needs KQUANT_TEST_GGUF_DIR with a small qwen3 dense GGUF).
Anti-vacuous by construction: prefill_step_size 2048 is a multiple of
the 256 cache allocation step, so a step-aligned prompt never strands a
strided slice - the alignment accident that hid issue #104 from every
single-stream run. The prompt is padded until its token count is NOT a
multiple of 256 and the final prefill chunk is >= 8 tokens (the flash
arm threshold).
"""

import pytest

from gmlx.gen.generation import generate


def _find_small_dense(gguf_dir):
    for path in sorted(gguf_dir.rglob("*.gguf")):
        name = path.name.lower()
        if "qwen3-0.6b" in name and "mtp" not in name:
            return path
    pytest.skip("no qwen3-0.6b GGUF under KQUANT_TEST_GGUF_DIR")


def test_kv8_long_prompt_strided_prefill(gguf_dir, capsys):
    from gmlx.load.loader import load_model

    model, config, tok = load_model(str(_find_small_dense(gguf_dir)))
    needle = "The vault code is INDIGO4471."
    filler = ("The quick brown fox jumps over the lazy dog. "
              "Nothing else in this text matters. ")
    body = needle + " " + filler * 90
    question = " What is the vault code? Answer with the code only."

    def _tokens(text):
        msgs = [{"role": "user", "content": text}]
        s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
        return len(tok.encode(s))

    # Land in (2048 + 8, next) with count % 256 in [8, 248]: the last
    # prefill chunk crosses into a quantized cache as a strided slice.
    for _ in range(40):
        n = _tokens(body + question)
        if n > 2056 and 8 <= n % 256 <= 248:
            break
        body += filler
    else:
        pytest.fail(f"could not shape an anti-vacuous prompt (n={n})")

    text = generate(
        model, tok, body + question,
        max_tokens=200, temp=0.0,
        kv_bits=8, kv_group_size=64, quantized_kv_start=0,
        template_kwargs={"enable_thinking": False},
    )
    err = capsys.readouterr().err
    assert "quantized 27/28 attn layers" in err
    assert "INDIGO4471" in text, f"needle lost: {text[:200]!r}"
    # The issue #104 signature was a repeating char loop.
    assert not any(ch * 40 in text for ch in set(text[:400])), "char loop"
