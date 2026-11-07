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


def test_a_json_parse_leaves_the_event_loop_free(monkeypatch):
    """The price, the scan and the parse of a JSON body run in a worker
    thread: the loop serves other requests while a large body parses."""
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()
    real = bi._loads

    def parse(box, free):
        started.set()
        assert release.wait(10), "the loop never ran beside the parse"
        return real(box, free)

    async def main():
        task = asyncio.create_task(
            bi.read_json(_json_request(b'{"prompt": "a cat"}'), 1 << 30))
        await asyncio.to_thread(started.wait, 10)
        # Runs on the loop, so only while the parse leaves it free.
        release.set()
        return await task
    monkeypatch.setattr(bi, "_loads", parse)
    assert asyncio.run(main()) == {"prompt": "a cat"}


def _stuck_request(head: bytes, length: int):
    """A JSON request with Content-Length ``length`` that sends ``head``,
    then nothing, and an Event set once its read waits past ``head``."""
    import asyncio

    from starlette.requests import Request

    waiting = asyncio.Event()
    parts = [head]

    async def receive():
        if parts:
            return {"type": "http.request", "body": parts.pop(),
                    "more_body": True}
        waiting.set()
        await asyncio.Event().wait()
    request = Request({"type": "http", "method": "POST", "path": "/v1/videos",
                       "query_string": b"", "headers": [
                           (b"content-type", b"application/json"),
                           (b"content-length", str(length).encode())]},
                      receive)
    return request, waiting


def test_a_json_body_that_stops_sending_holds_only_the_bytes_it_sent():
    """A client that sends a 64 MiB Content-Length and 10 bytes holds
    those bytes, not a buffer of its length, while the read waits."""
    import asyncio
    import tracemalloc

    async def main():
        request, waiting = _stuck_request(b'{"prompt":', 64 << 20)
        task = asyncio.create_task(bi.read_json(request, 1 << 30))
        await waiting.wait()
        held = tracemalloc.get_traced_memory()[0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return held
    tracemalloc.start()
    try:
        held = asyncio.run(main())
    finally:
        tracemalloc.stop()
    assert held < 1 << 20, held


def test_a_json_body_that_stops_sending_is_ended_with_a_408(monkeypatch):
    """Each chunk of a JSON body waits STALL_S seconds at most, as a
    form's does: past it, the read ends with a 408."""
    import asyncio

    monkeypatch.setattr(bi, "STALL_S", 0.05)

    async def main():
        request, _ = _stuck_request(b'{"prompt":', 1000)
        return await asyncio.wait_for(bi.read_json(request, 1 << 30), 10)
    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(main())
    assert e.value.status == 408
    assert str(e.value) == (
        "the request body sent nothing for 0.05 s with 1.0 KiB of it still "
        "to come, so the server ended the request; send it again")
    assert e.value.headers == {"Connection": "close"}


def test_a_chunked_form_file_past_the_memory_free_stops_as_it_passes():
    """An images form with no Content-Length is refused once the bytes
    arrived pass the memory free, as a Content-Length past it is, before
    the rest of its file is spooled."""
    import asyncio

    from starlette.requests import Request

    b = b"gmlxboundary"
    head = (b"--" + b + b'\r\nContent-Disposition: form-data; name="image";'
            b' filename="a.png"\r\nContent-Type: image/png\r\n\r\n')
    free, chunk, total = 4 << 20, 64 << 10, 32 << 20
    sent = [0]

    async def receive():
        if sent[0] == 0:
            sent[0] = len(head)
            return {"type": "http.request", "body": head, "more_body": True}
        sent[0] += chunk
        return {"type": "http.request", "body": b"x" * chunk,
                "more_body": sent[0] < total}
    request = Request({"type": "http", "method": "POST",
                       "path": "/v1/images/edits", "query_string": b"",
                       "headers": [(b"content-type",
                                    b"multipart/form-data; boundary=" + b)]},
                      receive)
    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(bi.read_form_in_ram(request, free))
    assert e.value.status == 413
    assert str(e.value).startswith("the request body is over 4.0 MiB, and "
                                   "4.0 MiB of memory is free"), e.value
    assert sent[0] <= free + len(head) + chunk
