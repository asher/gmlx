"""gmlx/serve/patches/body_intake.py: the figure a route prices its own
body against, the bytes free beside what the pool holds."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gmlx.serve.patches import _common
from gmlx.serve.patches import body_intake as bi


def test_the_free_figure_is_the_budget_less_the_resident_and_incoming(
        monkeypatch):
    """The resident bytes and the upload bodies admitted both count, and
    GMLX_OVERCOMMIT or a pool with no budget leaves the body unpriced."""
    figures = SimpleNamespace(budget=100 << 20, resident=30 << 20,
                              incoming=20 << 20)
    monkeypatch.setattr(_common, "_get_pool",
                        lambda: SimpleNamespace(figures=figures))
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    assert bi.free_bytes() == 50 << 20
    figures.incoming = 90 << 20
    assert bi.free_bytes() == 0
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert bi.free_bytes() is None
    monkeypatch.delenv("GMLX_OVERCOMMIT")
    figures.budget = 0
    assert bi.free_bytes() is None


def test_a_refusal_rounds_the_need_up_and_the_figure_down():
    """A body one byte past the figure never reads as the figure."""
    assert bi._mib((1 << 30) + 1, True) == "1.01 GiB"
    assert bi._mib((1 << 30) - 1) == "1023.9 MiB"
    assert str(bi._over_free((1 << 30) + 1, 1 << 30, True)).startswith(
        "the request body is 1.01 GiB, and 1.00 GiB of memory is free")
    assert str(bi._over_room((8 << 30) + 100_000, 8 << 30)).startswith(
        "the files of this job are 8.01 GiB, and the video job store has "
        "8.00 GiB of room left;")
    assert bi._mib((2 << 20) - 1) == "1.9 MiB"
    assert bi._mib((2 << 20) - 1, True) == "2.0 MiB"
    assert bi._mib(1023, True) == "1.0 KiB"


def test_a_slow_form_waits_afresh_for_each_chunk(tmp_path, monkeypatch):
    """A form that keeps sending is read to its end however long it takes:
    the wait for its next chunk starts again at each one."""
    import asyncio

    from starlette.requests import Request

    b = b"gmlxboundary"
    body = (b"--" + b + b'\r\nContent-Disposition: form-data; name="prompt"'
            b"\r\n\r\na cat\r\n--" + b + b'\r\nContent-Disposition: '
            b'form-data; name="first_frame"; filename="f.png"\r\n'
            b"Content-Type: image/png\r\n\r\n" + b"x" * 300 + b"\r\n--" + b
            + b"--\r\n")
    parts = [body[i:i + 100] for i in range(0, len(body), 100)]

    async def receive():
        chunk = parts.pop(0)
        return {"type": "http.request", "body": chunk,
                "more_body": bool(parts)}

    request = Request({"type": "http", "method": "POST", "headers": [
        (b"content-type", b"multipart/form-data; boundary=" + b),
        (b"content-length", str(len(body)).encode())]}, receive)
    waits = []

    async def within(timeout, aw):
        waits.append(timeout)
        return await aw
    monkeypatch.setattr(bi, "_within", within)

    async def take(n):
        return True, n, 0
    fields, files = asyncio.run(bi.read_form_to_disk(
        request, str(tmp_path / "in"), take, free=None))
    assert fields == [("prompt", "a cat")] and list(files) == ["first_frame"]
    # A full wait for each chunk, and for the end of the stream.
    assert len(waits) > len(body) // 100
    assert set(waits) == {bi.STALL_S}


def _json_request(body: bytes, ctype: str = "application/json"):
    from starlette.requests import Request

    chunks = [body[i:i + (64 << 10)] for i in range(0, len(body), 64 << 10)]

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0),
                    "more_body": bool(chunks)}
        return {"type": "http.disconnect"}
    return Request({"type": "http", "method": "POST", "path": "/v1/videos",
                    "query_string": b"", "headers": [
                        (b"content-type", ctype.encode()),
                        (b"content-length", str(len(body)).encode())]},
                   receive)


@pytest.mark.parametrize("body", [
    '{"prompt": "un caf\u00e9 \u4e00 \U0001F600", "seed": 1}'.encode(),
    # A body that holds a \x or \U of its own takes the escape's slow way.
    '{"prompt": "C:\\\\xampp \u00e9 \\\\U", "x": [1.5, null]}'.encode(),
    # A surrogate encoded as UTF-8 reads as json.loads reads it.
    b'{"prompt": "a \xed\xa0\x80 b \xc3\xa9"}',
    b'\xef\xbb\xbf{"prompt": "\xc3\xa9"}',
    '{"prompt": "\u00e9\U0001F600"}'.encode("utf-16"),
])
def test_a_json_body_reads_as_json_loads_reads_it(body):
    """The body is escaped to ASCII or decoded as it is, whichever holds
    less, and either way parses to what json.loads makes of its bytes."""
    import asyncio

    assert asyncio.run(bi.read_json(_json_request(body), 1 << 30)) == \
        json.loads(body)


def test_a_json_body_that_is_not_utf8_is_refused_as_json_loads_refuses_it():
    import asyncio

    body = b'{"prompt": "caf\xe9 au lait \xc3\xa9"}'
    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(bi.read_json(_json_request(body), 1 << 30))
    with pytest.raises(UnicodeDecodeError) as want:
        json.loads(body)
    assert e.value.status == 400
    assert str(e.value) == f"the request body is not valid JSON ({want.value})"
