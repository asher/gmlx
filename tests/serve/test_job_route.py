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


def test_d1_decisions_take_one_server_turn_in_arrival_order():
    order = []

    async def one(name, gate):
        async with job_route.decision_turn(None, 600):
            order.append(name)
            assert job_route.decision_turn_held()
            await gate.wait()

    async def main():
        gates = {n: asyncio.Event() for n in "ab"}
        tasks = []
        for n in "ab":
            tasks.append(asyncio.create_task(one(n, gates[n])))
            await _settle()
        assert order == ["a"]
        gates["a"].set()
        await _settle()
        assert order == ["a", "b"]
        gates["b"].set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert not job_route.decision_turn_held()
    assert job_route._TURNS == {}


def test_a_d1_wait_past_its_bound_is_busy_and_keeps_the_holder():
    async def main():
        hold = asyncio.Event()
        ran = []

        async def first():
            async with job_route.decision_turn(None, 600):
                ran.append("first")
                await hold.wait()
                ran.append("first done")

        task = asyncio.create_task(first())
        await _settle()
        with pytest.raises(job_route.TurnBusy):
            async with job_route.decision_turn(None, 0):
                ran.append("busy")
        assert job_route.decision_turn_held()
        hold.set()
        await task
        assert ran == ["first", "first done"]
        async with job_route.decision_turn(None, 0):
            ran.append("after")
        assert ran[-1] == "after"

    asyncio.run(main())
    assert job_route._TURNS == {}


def test_a_d1_waiter_that_disconnects_leaves_and_the_next_one_runs():
    async def main():
        hold, gone = asyncio.Event(), asyncio.Event()
        order = []

        async def first():
            async with job_route.decision_turn(None, 600):
                order.append("first")
                await hold.wait()

        async def leaver():
            watcher = asyncio.create_task(gone.wait())
            with pytest.raises(JobCancelled):
                async with job_route.decision_turn(watcher, 600):
                    order.append("leaver")

        async def third():
            async with job_route.decision_turn(None, 600):
                order.append("third")

        tasks = [asyncio.create_task(first())]
        await _settle()
        tasks += [asyncio.create_task(leaver())]
        await _settle()
        tasks += [asyncio.create_task(third())]
        await _settle()
        gone.set()
        await _settle()
        hold.set()
        await asyncio.gather(*tasks)
        assert order == ["first", "third"]

    asyncio.run(main())
    assert job_route._TURNS == {}


def test_a_failed_d1_job_gives_the_turn_back():
    async def main():
        with pytest.raises(RuntimeError):
            async with job_route.decision_turn(None, 600):
                raise RuntimeError("the job failed")
        async with job_route.decision_turn(None, 0):
            pass

    asyncio.run(main())
    assert job_route._TURNS == {}
