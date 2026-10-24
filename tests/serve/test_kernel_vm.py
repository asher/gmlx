"""Kernel memory readers: the free-page stall watch."""

import gmlx.serve.kernel_vm as kv


def test_a_stall_is_waiters_with_no_reclaim(monkeypatch):
    # Threads waiting while the kernel compresses, or while fewer of them
    # wait, are reclaim at work; the clock runs only once both stop.
    box = {"waiters": 3, "marks": (0, 0)}
    monkeypatch.setattr(kv, "free_page_waiters", lambda: box["waiters"])
    monkeypatch.setattr(kv, "reclaim_marks", lambda: box["marks"])
    watch = kv.StarveWatch()
    assert watch() == 0.0
    watch.since -= 1.0
    assert watch() >= 1.0
    box["marks"] = (1 << 30, 0)
    assert watch() < 0.1
    watch.since -= 1.0
    box["waiters"] = 2
    assert watch() < 0.1
    box["waiters"] = 0
    assert watch() == 0.0 and watch.since is None
