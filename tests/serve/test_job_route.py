"""The shared job-route pieces: the per-engine scoring turn, its waiter
count, and the disconnect watcher. No model is loaded."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.serve.queue_cap as qc  # noqa: E402
from gmlx.serve import job_route  # noqa: E402
from gmlx.serve.engine_jobs import JobCancelled  # noqa: E402


def _rg():
    return SimpleNamespace(requests=SimpleNamespace(qsize=lambda: 0))


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def test_a_disconnect_sets_the_stop_event():
    class _Req:
        async def is_disconnected(self):
            return True

    stop = threading.Event()
    asyncio.run(job_route.watch_disconnect(_Req(), stop))
    assert stop.is_set()


def test_turns_go_in_arrival_order():
    rg = _rg()
    order = []

    async def one(name, gate):
        async with job_route.scoring_turn(rg):
            order.append(name)
            await gate.wait()

    async def main():
        gates = {n: asyncio.Event() for n in "abc"}
        tasks = []
        for n in "abc":
            tasks.append(asyncio.create_task(one(n, gates[n])))
            await _settle()
        assert order == ["a"]
        assert job_route.scoring_waiters(rg) == 2
        for n in "abc":
            gates[n].set()
        await asyncio.gather(*tasks)
        assert job_route.scoring_waiters(rg) == 0

    asyncio.run(main())
    assert order == ["a", "b", "c"]
    assert job_route._TURNS == {}


def test_a_disconnect_while_waiting_drops_the_wait():
    rg = _rg()
    order = []

    async def main():
        hold = asyncio.Event()
        gone = asyncio.Event()

        async def first():
            async with job_route.scoring_turn(rg):
                order.append("first")
                await hold.wait()

        async def leaver():
            watcher = asyncio.create_task(gone.wait())
            with pytest.raises(JobCancelled):
                async with job_route.scoring_turn(rg, watcher):
                    order.append("leaver")

        async def third():
            async with job_route.scoring_turn(rg):
                order.append("third")

        tasks = [asyncio.create_task(first())]
        await _settle()
        tasks.append(asyncio.create_task(leaver()))
        await _settle()
        tasks.append(asyncio.create_task(third()))
        await _settle()
        assert job_route.scoring_waiters(rg) == 2
        gone.set()
        await _settle()
        assert job_route.scoring_waiters(rg) == 1
        hold.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert order == ["first", "third"]
    assert job_route._TURNS == {}


def test_an_error_in_the_turn_frees_it():
    rg = _rg()

    async def main():
        with pytest.raises(ValueError):
            async with job_route.scoring_turn(rg):
                raise ValueError
        async with job_route.scoring_turn(rg):
            pass

    asyncio.run(main())
    assert job_route._TURNS == {}


def test_the_turn_keys_on_the_engine_behind_the_guard():
    rg = _rg()
    guard = SimpleNamespace(_rg=rg)

    async def main():
        hold = asyncio.Event()

        async def first():
            async with job_route.scoring_turn(guard):
                await hold.wait()

        async def second():
            async with job_route.scoring_turn(rg):
                pass

        tasks = [asyncio.create_task(first())]
        await _settle()
        tasks.append(asyncio.create_task(second()))
        await _settle()
        assert job_route.scoring_waiters(rg) == 1
        assert job_route.scoring_waiters(guard) == 1
        hold.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())


def test_scoring_waiters_count_toward_the_queue_depth(monkeypatch):
    monkeypatch.setattr(qc, "_GEN_REF", lambda: None)
    rg = _rg()

    async def main():
        hold = asyncio.Event()

        async def first():
            async with job_route.scoring_turn(rg):
                await hold.wait()

        async def waiter():
            async with job_route.scoring_turn(rg):
                pass

        tasks = [asyncio.create_task(first())]
        await _settle()
        tasks += [asyncio.create_task(waiter()), asyncio.create_task(waiter())]
        await _settle()
        depth = qc._waiting_depth(rg)
        hold.set()
        await asyncio.gather(*tasks)
        return depth

    assert asyncio.run(main()) == 2
    assert qc._waiting_depth(rg) == 0
