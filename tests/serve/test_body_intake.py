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


@pytest.mark.parametrize("to_disk", [True, False])
def test_a_slow_form_waits_afresh_for_each_chunk(tmp_path, monkeypatch,
                                                 to_disk):
    """A video form, or an images form, that keeps sending is read to its
    end however long it takes: the wait for its next chunk starts again
    at each one."""
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

    request = Request({"type": "http", "method": "POST",
                       "path": "/v1/videos" if to_disk else "/v1/images/edits",
                       "query_string": b"", "headers": [
                           (b"content-type",
                            b"multipart/form-data; boundary=" + b),
                           (b"content-length", str(len(body)).encode())]},
                      receive)
    waits = []

    async def within(timeout, aw):
        waits.append(timeout)
        return await aw
    monkeypatch.setattr(bi, "_within", within)

    async def take(n):
        return True, n, 0

    async def in_ram():
        form = await bi.read_form_in_ram(request, None)
        try:
            items = list(form.multi_items())
            return ([(k, v) for k, v in items if isinstance(v, str)],
                    {k: v for k, v in items if not isinstance(v, str)})
        finally:
            await form.close()
    fields, files = asyncio.run(bi.read_form_to_disk(
        request, str(tmp_path / "in"), take, free=None) if to_disk
        else in_ram())
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
    # A surrogate encoded as UTF-8 reads as json.loads reads it, and so
    # does a pair of them, or one before the \u escape of the other.
    b'{"prompt": "a \xed\xa0\x80 b \xc3\xa9"}',
    b'{"prompt": "a \xed\xa0\xbd\xed\xb8\x80 \xc3\xa9"}',
    b'{"prompt": "\xed\xa0\xbd\\ude00 \xc3\xa9"}',
    b'\xef\xbb\xbf{"prompt": "\xc3\xa9"}',
    '{"prompt": "\u00e9\U0001F600"}'.encode("utf-16"),
])
def test_a_json_body_reads_as_json_loads_reads_it(body):
    """The body is escaped to ASCII or decoded as it is, whichever holds
    less, and either way parses to what json.loads makes of its bytes."""
    import asyncio

    assert asyncio.run(bi.read_json(_json_request(body), 1 << 30)) == \
        json.loads(body)


@pytest.mark.parametrize("body", [
    b'{"prompt": "caf\xe9 au lait \xc3\xa9"}',
    # A backslash before a character that is not ASCII escapes nothing.
    '{"prompt": "\\\U0001F600"}'.encode(),
    '{"prompt": "a \\\\\\\u00e9"}'.encode(),
])
def test_a_json_body_json_loads_refuses_is_refused_with_its_error(body):
    import asyncio

    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(bi.read_json(_json_request(body), 1 << 30))
    with pytest.raises(ValueError) as want:
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

    def parse(*a):
        started.set()
        assert release.wait(10), "the loop never ran beside the parse"
        return real(*a)

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


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_a_str_widened_twice_parses_within_its_price(ensure_ascii):
    """A str whose parse widens it to 2 bytes a character, then to 4, holds
    each narrower copy as it widens: its read and parse peak within the
    price."""
    import asyncio
    import tracemalloc

    prompt = "中" + "a" * (2 << 20) + "\U0001F600"
    body = json.dumps({"prompt": prompt}, ensure_ascii=ensure_ascii).encode()
    need = bi.json_price(body)
    request = _json_request(body)
    tracemalloc.start()
    try:
        doc = asyncio.run(bi.read_json(request, need))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert doc == {"prompt": prompt}
    assert peak <= need, peak / need


def _form_request(path: str, body: bytes, ctype: str):
    """A request whose body arrives in 64 KiB chunks, as a socket's does."""
    from starlette.requests import Request

    chunks = [body[i:i + (64 << 10)] for i in range(0, len(body), 64 << 10)]

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0),
                    "more_body": bool(chunks)}
        return {"type": "http.disconnect"}
    return Request({"type": "http", "method": "POST", "path": path,
                    "query_string": b"", "headers": [
                        (b"content-type", ctype.encode()),
                        (b"content-length", str(len(body)).encode())]},
                   receive)


def test_a_data_url_in_an_edit_forms_text_field_is_read_within_its_price():
    """The gate decodes an edit form's image sent as a data URL in a text
    field: what the read and the checks hold at their peak never passes
    the price, so one byte less is refused."""
    import asyncio
    import base64
    import os
    import tracemalloc

    b = "gmlxboundary"
    url = ("data:image/png;base64,"
           + base64.b64encode(os.urandom(2 << 20)).decode())
    body = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
            f'a cat\r\n--{b}\r\nContent-Disposition: form-data; '
            f'name="image"\r\n\r\n{url}\r\n--{b}--\r\n').encode()
    ctype = f"multipart/form-data; boundary={b}"

    async def read(free):
        form = await bi.read_form_in_ram(
            _form_request("/v1/images/edits", body, ctype), free)
        await form.close()
    # A first read of the same body loads what a read loads once per
    # process, such as Pillow's format plugins; the price covers a read,
    # so the second read is the one measured.
    asyncio.run(read(1 << 40))
    tracemalloc.start()
    try:
        asyncio.run(read(1 << 40))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(read(peak - 1))
    assert e.value.status == 413


class _Turn:
    """A parse turn that sets ``parked`` when a body waits on it while
    another holds it."""

    def __init__(self):
        import asyncio
        import threading

        self.sem = asyncio.Semaphore(1)
        self.parked = threading.Event()

    def locked(self):
        return self.sem.locked()

    async def acquire(self):
        if self.sem.locked():
            self.parked.set()
        return await self.sem.acquire()

    def release(self):
        self.sem.release()


def _own_turn(monkeypatch) -> _Turn:
    """The parse turn of the next event loop, recorded."""
    turn = _Turn()

    class Turns(dict):
        def get(self, loop, default=None):
            return turn
    monkeypatch.setattr(bi, "_TURNS", Turns())
    return turn


def test_two_json_parses_take_their_turns(monkeypatch):
    """Each parse is priced against the same free figure, which books
    nothing: the second waits on the turn the first holds, and its parse
    starts only once the first ends, so their peaks never add up."""
    import asyncio
    import threading

    first, release = threading.Event(), threading.Event()
    second = threading.Event()
    real = bi._loads

    def parse(box, *a):
        if bytes(box[0]) == b'{"prompt": "a cat"}':
            first.set()
            assert release.wait(10), "the first parse was never let go"
        else:
            second.set()
        return real(box, *a)

    async def main():
        one = asyncio.create_task(
            bi.read_json(_json_request(b'{"prompt": "a cat"}'), 1 << 30))
        await asyncio.to_thread(first.wait, 10)
        two = asyncio.create_task(
            bi.read_json(_json_request(b'{"prompt": "a dog"}'), 1 << 30))
        # Until the second body parks on the turn, or its parse starts.
        for _ in range(1_000_000):
            if turn.parked.is_set() or second.is_set():
                break
            await asyncio.sleep(0)
        parked, started = turn.parked.is_set(), second.is_set()
        release.set()
        return parked, started, await one, await two
    turn = _own_turn(monkeypatch)
    monkeypatch.setattr(bi, "_loads", parse)
    parked, started, one, two = asyncio.run(main())
    assert parked and not started, \
        "the second parse started beside the first"
    assert (one, two) == ({"prompt": "a cat"}, {"prompt": "a dog"})


def test_short_strs_after_the_last_long_one_are_priced_as_text():
    """The strs of a body after its last long str price as the text around
    them, wherever that long str is: only a str with no closing quote is
    priced as one str to the end of the body."""
    short = ",".join(['"a\\n"'] * 30000)
    long_ = '"' + "x" * 4000 + '"'
    first = f'{{"a": {long_}, "b": [{short}]}}'.encode()
    last = f'{{"b": [{short}], "a": {long_}}}'.encode()
    assert len(first) == len(last)
    assert bi.json_price(first) == bi.json_price(last)
    assert bi.json_price(f'{{"b": [{short}]}}'.encode()) < \
        bi.json_price(first)
    # A str with no closing quote is made up to the end of the body.
    cut = first[:-30000]
    assert bi.json_price(cut + b'"' + b"x" * 30000) > bi.json_price(cut)


def _cut_form(file_field: str) -> tuple[bytes, str]:
    """A form whose body ends inside its file, with no closing boundary."""
    b = "gmlxboundary"
    body = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
            f'a cat\r\n--{b}\r\nContent-Disposition: form-data; '
            f'name="{file_field}"; filename="a.png"\r\nContent-Type: '
            f"image/png\r\n\r\n").encode() + b"x" * 3000
    return body, f"multipart/form-data; boundary={b}"


def test_a_cut_images_form_closes_the_file_it_began(monkeypatch):
    """A form whose body ends inside a file reads its whole parts; the
    file it began is closed, not left open to the end of the process."""
    import asyncio
    import tempfile

    from starlette import formparsers

    made: list = []

    class Spool(tempfile.SpooledTemporaryFile):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Spool)
    body, ctype = _cut_form("image")

    async def read():
        form = await bi.read_form_in_ram(
            _form_request("/v1/images/edits", body, ctype), 1 << 30)
        items = list(form.multi_items())
        await form.close()
        return items
    items = asyncio.run(read())
    assert items == [("prompt", "a cat")]
    assert made and all(f.closed for f in made)


def test_a_cut_video_form_leaves_no_part_file(tmp_path):
    """A video form whose body ends inside a file removes the part it
    began, and keeps no handle on it."""
    import asyncio
    import gc
    import warnings

    body, ctype = _cut_form("first_frame")

    async def take(n):
        return True, n, 0
    folder = tmp_path / "in"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        fields, files = asyncio.run(bi.read_form_to_disk(
            _form_request("/v1/videos", body, ctype), str(folder), take,
            free=None))
        gc.collect()
    assert fields == [("prompt", "a cat")] and files == {}
    assert list(folder.iterdir()) == []


def test_an_images_form_that_stops_sending_is_ended_with_a_408(monkeypatch):
    """Each chunk of an images form waits STALL_S seconds at most, as a
    JSON body's does: past it, the read ends with a 408."""
    import asyncio

    from starlette.requests import Request

    monkeypatch.setattr(bi, "STALL_S", 0.05)
    body, ctype = _cut_form("image")
    parts = [body]

    async def receive():
        if parts:
            return {"type": "http.request", "body": parts.pop(),
                    "more_body": True}
        await asyncio.Event().wait()
    length = len(body) + 1000
    request = Request({"type": "http", "method": "POST",
                       "path": "/v1/images/edits", "query_string": b"",
                       "headers": [(b"content-type", ctype.encode()),
                                   (b"content-length", str(length).encode())]},
                      receive)

    async def main():
        return await asyncio.wait_for(bi.read_form_in_ram(request, 1 << 30),
                                      10)
    with pytest.raises(bi.BodyRefused) as e:
        asyncio.run(main())
    assert e.value.status == 408
    assert str(e.value) == (
        "the request body sent nothing for 0.05 s with 1.0 KiB of it still "
        "to come, so the server ended the request; send it again")
    assert e.value.headers == {"Connection": "close"}


def test_a_json_body_sent_in_small_pieces_holds_its_bytes_not_theirs():
    """A client that sends its body two bytes at a time makes a bytes
    object of each piece: the read holds the body's bytes, within the
    price of its read, not each piece's object."""
    import asyncio
    import tracemalloc

    from starlette.requests import Request

    n = 128 << 10
    sent = [0]

    async def receive():
        # Each piece made as it arrives, as a socket's are, with the loop
        # run between them.
        await asyncio.sleep(0)
        piece = bytes((120, 120))
        sent[0] += 2
        return {"type": "http.request", "body": piece,
                "more_body": sent[0] < n}
    request = Request({"type": "http", "method": "POST", "path": "/v1/videos",
                       "query_string": b"", "headers": [
                           (b"content-type", b"application/json"),
                           (b"content-length", str(n).encode())]}, receive)
    tracemalloc.start()
    try:
        buf = asyncio.run(bi._read_bytes(request, n, 1 << 30))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(buf) == n
    assert peak <= bi._min_price(n) + bi._SLACK, peak
