"""Reliability tests for RecurrentThread.Restart().

Thread/Future/timerfd are vendored verbatim from unitree_sdk2py; Restart() is
the local addition. These tests therefore concentrate on how Restart()
interacts with the inherited machinery: the Future state/Condition it silently
rebuilds, the threading.Thread it replaces, the __quit flag it clears, and the
timerfd the loop owns.

Everything is timing-based but deliberately loose: assertions are on
"did it happen at all within a deadline", never on exact call counts, except
where periodicity itself is under test.
"""

import errno
import os
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="timerfd_create is Linux-only"
)

from go2_deploy.utility import thread as thread_mod  # noqa: E402
from go2_deploy.utility.future import FutureResult  # noqa: E402
from go2_deploy.utility.thread import RecurrentThread  # noqa: E402

INTERVAL = 0.01  # 100 Hz; every test is built around this
DEADLINE = 2.0  # generous upper bound for "it should have happened by now"


def wait_for(pred, timeout=DEADLINE, poll=0.001):
    """Poll pred() until true or timeout. Returns the final value."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(poll)
    return bool(pred())


def is_alive(t):
    """Liveness of the underlying threading.Thread, read without going
    through the code under test."""
    return t._Thread__thread.is_alive()


def fd_count():
    return len(os.listdir("/proc/self/fd"))


class Recorder:
    """Loop target that records every invocation."""

    def __init__(self, sleep=0.0, raise_every=0):
        self._lock = threading.Lock()
        self.stamps = []
        self.seen = []  # (args, kwargs) per call
        self._sleep = sleep
        self._raise_every = raise_every

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.stamps.append(time.monotonic())
            self.seen.append((args, dict(kwargs)))
            n = len(self.stamps)
        if self._sleep:
            time.sleep(self._sleep)
        if self._raise_every and n % self._raise_every == 0:
            raise RuntimeError(f"target boom #{n}")
        return n

    @property
    def count(self):
        with self._lock:
            return len(self.stamps)

    def advanced(self, by=1):
        """Predicate factory: count has grown by `by` since now."""
        base = self.count
        return lambda: self.count >= base + by


@pytest.fixture
def fine_grained_switching():
    """Shrink the GIL switch interval from the 5 ms default.

    Race windows in Restart()/Wait() are a handful of bytecodes wide; at the
    default interval the notifying thread almost always runs to completion
    before the woken one is scheduled, so latent races never surface. This
    changes scheduling granularity only -- not semantics.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


@pytest.fixture
def make_thread():
    """Factory + guaranteed teardown.

    Teardown does not call Wait(): it pokes __quit directly so that a bug in
    Wait() cannot wedge the whole session, and it fails the test if a loop
    refuses to die (catches runaway/hot-spin loops).
    """
    created = []

    def _make(interval=INTERVAL, target=None, name=None, args=(), kwargs=None):
        t = RecurrentThread(
            interval=interval, target=target, name=name, args=args, kwargs=kwargs
        )
        created.append(t)
        return t

    yield _make

    leaked = []
    for t in created:
        t._RecurrentThread__quit = True
        t._Thread__thread.join(2.0)
        if is_alive(t):
            leaked.append(t._Thread__thread.name)
    assert not leaked, f"threads still alive after teardown: {leaked}"


# ---------------------------------------------------------------------------
# Group A -- baseline behaviour Restart() depends on
# ---------------------------------------------------------------------------


def test_loop_runs_periodically(make_thread):
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 10)
    t.Wait(DEADLINE)

    gaps = [b - a for a, b in zip(rec.stamps, rec.stamps[1:])]
    gaps.sort()
    median = gaps[len(gaps) // 2]
    assert INTERVAL * 0.5 < median < INTERVAL * 2.0, f"median gap {median}"


def test_wait_stops_the_loop(make_thread):
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 3)
    t.Wait(DEADLINE)

    frozen = rec.count
    time.sleep(INTERVAL * 5)
    assert rec.count == frozen


def test_wait_reports_whether_it_timed_out(make_thread):
    """Restart() is only legal once the previous run stopped, so the caller
    needs Wait() to tell it whether the wait actually completed.
    RecurrentThread.Wait() drops super().Wait()'s return value on the floor."""
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 2)
    assert t.Wait(DEADLINE) is True


# ---------------------------------------------------------------------------
# Group B -- Restart() core contract
# ---------------------------------------------------------------------------


def test_restart_after_wait_resumes_the_loop(make_thread):
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 3)
    t.Wait(DEADLINE)

    grew = rec.advanced(3)
    assert t.Restart() is True
    assert wait_for(grew), "loop did not resume after Restart()"
    t.Wait(DEADLINE)


def test_restart_installs_a_new_thread_and_keeps_the_name(make_thread):
    """Note: GetId() is NOT a run discriminator -- Linux recycles the tid of
    the exited thread, so the new run frequently reports the same ident."""
    t = make_thread(target=Recorder(), name="go2-lowcmd")
    t.Start()
    first_obj = t._Thread__thread
    t.Wait(DEADLINE)

    assert t.Restart() is True
    assert t._Thread__thread is not first_obj
    assert t._Thread__thread.name == "go2-lowcmd"
    assert is_alive(t)
    t.Wait(DEADLINE)


def test_restart_preserves_args_and_kwargs(make_thread):
    rec = Recorder()
    t = make_thread(target=rec, args=(1, 2), kwargs={"k": "v"})
    t.Start()
    assert wait_for(lambda: rec.count >= 2)
    t.Wait(DEADLINE)

    grew = rec.advanced(2)
    assert t.Restart() is True
    assert wait_for(grew)
    t.Wait(DEADLINE)

    assert rec.seen and all(call == ((1, 2), {"k": "v"}) for call in rec.seen), (
        f"args/kwargs not forwarded: {rec.seen[:3]}"
    )


def test_restart_preserves_the_interval(make_thread):
    rec = Recorder()
    t = make_thread(interval=INTERVAL, target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 3)
    t.Wait(DEADLINE)

    rec.stamps.clear()
    assert t.Restart() is True
    assert wait_for(lambda: rec.count >= 10)
    t.Wait(DEADLINE)

    gaps = sorted(b - a for a, b in zip(rec.stamps, rec.stamps[1:]))
    median = gaps[len(gaps) // 2]
    assert INTERVAL * 0.5 < median < INTERVAL * 2.0, f"median gap {median}"


def test_restart_fires_the_target_immediately(make_thread):
    """Characterization: __LoopFunc calls the target *before* its first timer
    read, and Restart() re-arms a fresh timerfd. So a restart re-fires straight
    away instead of preserving phase -- a supervisor cycling Wait()/Restart()
    faster than `interval` drives the target faster than its nominal rate,
    which for a lowcmd loop means back-to-back sends."""
    slow = 0.2  # long enough that "immediate" and "one period later" differ
    rec = Recorder()
    t = make_thread(interval=slow, target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 2)
    t.Wait(DEADLINE)
    assert wait_for(lambda: not is_alive(t))

    last_of_run_1 = rec.stamps[-1]
    grew = rec.advanced(1)
    restarted_at = time.monotonic()
    assert t.Restart() is True
    assert wait_for(grew)
    first_of_run_2 = rec.stamps[-1]
    t.Wait(DEADLINE)

    assert first_of_run_2 - restarted_at < slow / 2, (
        "expected an immediate re-fire after Restart(), got "
        f"{first_of_run_2 - restarted_at:.4f}s"
    )
    # And the flip side: the cycle costs roughly one whole period of silence,
    # because the old loop only notices __quit after its pending os.read()
    # returns. For a 500 Hz lowcmd loop that is one dropped command.
    assert first_of_run_2 - last_of_run_1 >= slow * 0.9, (
        f"restart seam unexpectedly short: {first_of_run_2 - last_of_run_1:.4f}s"
    )


def test_restart_while_running_is_refused_and_harmless(make_thread):
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 3)

    running_id = t.GetId()
    grew = rec.advanced(3)
    assert t.Restart() is False
    assert t.GetId() == running_id, "refused Restart() replaced the live thread"
    assert wait_for(grew), "refused Restart() disturbed the running loop"
    t.Wait(DEADLINE)


def test_restart_resets_the_future_state(make_thread):
    """Run 1 leaves the Future in READY. If Restart() did not reset it,
    GetResult() would return run 1's stale value immediately instead of
    tracking run 2."""
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 2)
    t.Wait(DEADLINE)

    assert t.Restart() is True
    started = time.monotonic()
    result = t.GetResult(0.2)
    elapsed = time.monotonic() - started

    assert result.code == FutureResult.FUTUTE_ERR_TIMEOUT
    assert elapsed >= 0.15, "GetResult() returned stale run-1 state instantly"
    t.Wait(DEADLINE)


def test_restart_before_any_start(make_thread):
    """A never-started RecurrentThread is not alive, so Restart() acts as a
    plain Start()."""
    rec = Recorder()
    t = make_thread(target=rec)

    assert t.Restart() is True
    assert wait_for(lambda: rec.count >= 3)
    t.Wait(DEADLINE)


def test_restart_survives_a_target_that_raises(make_thread):
    rec = Recorder(raise_every=1)
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 3)
    t.Wait(DEADLINE)

    grew = rec.advanced(3)
    assert t.Restart() is True
    assert wait_for(grew)
    t.Wait(DEADLINE)


def test_restart_after_the_loop_itself_crashed(make_thread):
    """A non-EAGAIN error out of os.read propagates and puts the Future in
    FAILED. Restart() must still be able to bring the loop back."""
    rec = Recorder()
    t = make_thread(target=rec)

    with _os_shim(read_error=OSError(errno.EIO, "injected")) as shim:
        t.Start()
        assert wait_for(lambda: not is_alive(t)), "loop should have died"
        assert t.GetResult(DEADLINE).code == FutureResult.FUTURE_ERR_FAILED
        del shim

    grew = rec.advanced(3)
    assert t.Restart() is True
    assert wait_for(grew), "Restart() did not recover from a crashed loop"
    t.Wait(DEADLINE)


def test_restart_on_the_zero_interval_path(make_thread):
    """interval<=0 selects __LoopFunc_0, which Restart() re-selects. That path
    must call the target with the configured args, before and after Restart."""
    rec = Recorder(sleep=0.001)
    t = make_thread(interval=0.0, target=rec, args=(7,), kwargs={"k": 9})

    # Silenced: when this path is broken it spins at full speed printing one
    # traceback summary per iteration, which buries the pytest report.
    with _silence_prints():
        t.Start()
        called = wait_for(lambda: rec.count >= 3, timeout=0.3)
        t.Wait(DEADLINE)
        assert called, "zero-interval loop never called the target"

        grew = rec.advanced(3)
        assert t.Restart() is True
        resumed = wait_for(grew, timeout=0.3)
        t.Wait(DEADLINE)
    assert resumed, "zero-interval loop did not resume after Restart()"

    assert all(call == ((7,), {"k": 9}) for call in rec.seen), (
        f"args/kwargs not forwarded: {rec.seen[:3]}"
    )


# ---------------------------------------------------------------------------
# Group C -- races between Restart(), Wait() and the loop
# ---------------------------------------------------------------------------


def test_wait_implies_the_thread_is_dead(make_thread, fine_grained_switching):
    """The documented Restart() precondition. Wait() unblocks on the Future's
    condition, which Thread.__ThreadFunc notifies from *inside* the thread --
    so returning from Wait() does not by itself mean the thread has exited."""
    rec = Recorder()
    t = make_thread(target=rec)

    alive_after_wait = []
    for cycle in range(50):
        assert (t.Start() if cycle == 0 else t.Restart()) is not False
        assert wait_for(lambda: rec.count >= 1)
        t.Wait(DEADLINE)
        if is_alive(t):
            alive_after_wait.append(cycle)
            wait_for(lambda: not is_alive(t))  # so the next Restart() can work

    assert not alive_after_wait, (
        f"thread still alive immediately after Wait() in {len(alive_after_wait)}/50 "
        f"cycles: {alive_after_wait}"
    )


def test_wait_with_a_short_timeout_leaves_the_thread_running(make_thread):
    """__quit is only polled after the blocking os.read() returns, so shutdown
    latency is up to one full interval no matter what timeout Wait() is given.
    Wait() returning is therefore not sufficient to make Restart() legal, and
    because Wait() returns None the caller cannot detect the difference."""
    slow = 0.4
    rec = Recorder()
    t = make_thread(interval=slow, target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 1)

    t.Wait(0.02)  # far shorter than the interval
    assert is_alive(t), "test precondition: loop should still be in os.read()"
    assert t.Restart() is False, "Restart() ran while the old loop was alive"

    assert wait_for(lambda: not is_alive(t), timeout=slow * 3)
    assert t.Restart() is True


def test_repeated_restart_cycles(make_thread, fine_grained_switching):
    """The field usage pattern: stop the control loop, restart it, many times.
    Every Wait()/Restart() pair must succeed -- a supervisor has no other
    handshake available."""
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()

    refused = []
    for cycle in range(50):
        assert wait_for(rec.advanced(2)), f"loop stalled in cycle {cycle}"
        t.Wait(DEADLINE)
        if not t.Restart():
            refused.append(cycle)
            assert wait_for(lambda: not is_alive(t))
            assert t.Restart(), f"unrecoverable in cycle {cycle}"

    t.Wait(DEADLINE)
    assert not refused, (
        f"Restart() refused after Wait() in {len(refused)}/50 cycles: {refused}"
    )


def test_concurrent_restarts_start_exactly_one_loop(
    make_thread, fine_grained_switching
):
    """Restart() reads is_alive(), then rebuilds and Starts, with no lock. Two
    callers can both observe a dead thread and both build a loop -- the first
    one is then orphaned (it keeps running, unreachable and unstoppable)."""
    rec = Recorder()
    t = make_thread(target=rec)

    lock = threading.Lock()
    bad_rounds = []

    for round_no in range(25):
        barrier = threading.Barrier(4)
        outcomes = []

        def racer():
            barrier.wait()
            try:
                outcome = t.Restart()
            except RuntimeError as exc:  # threads can only be started once
                outcome = exc
            with lock:
                outcomes.append(outcome)

        racers = [threading.Thread(target=racer, daemon=True) for _ in range(4)]
        for r in racers:
            r.start()
        for r in racers:
            r.join(DEADLINE)

        raised = [o for o in outcomes if isinstance(o, BaseException)]
        started = [o for o in outcomes if o is True]
        if raised or len(started) != 1:
            bad_rounds.append((round_no, len(started), [repr(e) for e in raised]))

        t.Wait(DEADLINE)
        assert wait_for(lambda: not is_alive(t))

    assert not bad_rounds, (
        "concurrent Restart() did not serialize (round, #started, exceptions): "
        f"{bad_rounds}"
    )


def test_restart_keeps_the_same_condition_object(make_thread):
    """Restart() goes through Thread.__init__ -> Future.__init__, which
    allocates a brand-new Condition. Any thread already blocked in Wait() on
    the old Condition can never be notified by the new run."""
    t = make_thread(target=Recorder())
    t.Start()
    before = t._Future__condition
    t.Wait(DEADLINE)

    assert t.Restart() is True
    assert t._Future__condition is before
    t.Wait(DEADLINE)


def test_waiter_in_another_thread_is_not_stranded_by_restart(make_thread):
    """Practical form of the Condition-swap hazard: a supervisor thread parked
    in Wait() while the main thread cycles the loop must always wake up."""
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()

    stranded = []

    for cycle in range(20):
        assert wait_for(rec.advanced(1))

        done = threading.Event()

        def waiter():
            t.Wait(DEADLINE)
            done.set()

        w = threading.Thread(target=waiter, daemon=True)
        w.start()
        if not done.wait(DEADLINE * 1.5):
            stranded.append(cycle)
        w.join(0.5)

        if not t.Restart():
            time.sleep(INTERVAL * 3)
            assert t.Restart(), f"could not restart in cycle {cycle}"

    t.Wait(DEADLINE)
    assert not stranded, f"Wait() never returned in cycles {stranded}"


# ---------------------------------------------------------------------------
# Group D -- resource hygiene across restarts
# ---------------------------------------------------------------------------


def test_no_fd_leak_across_restarts(make_thread):
    """Each run of __LoopFunc creates a timerfd; each must be closed."""
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    assert wait_for(lambda: rec.count >= 2)
    t.Wait(DEADLINE)

    baseline = fd_count()
    for cycle in range(20):
        assert t.Restart() is True, f"Restart() refused in cycle {cycle}"
        assert wait_for(rec.advanced(1))
        t.Wait(DEADLINE)
    assert wait_for(lambda: not is_alive(t))

    assert fd_count() - baseline <= 2, (
        f"fd count grew {baseline} -> {fd_count()} over 20 restarts"
    )


def test_timerfd_is_closed_when_the_loop_crashes(make_thread):
    """The os.read error path re-raises without closing the timerfd, so a
    supervisor that keeps calling Restart() leaks one fd per crash."""
    rec = Recorder()
    t = make_thread(target=rec)

    with _os_shim(read_error=OSError(errno.EIO, "injected")) as shim:
        t.Start()
        assert wait_for(lambda: not is_alive(t))
        assert shim.opened, "loop never created a timerfd"
        assert shim.closed == shim.opened, (
            f"timerfd(s) {set(shim.opened) - set(shim.closed)} leaked on crash"
        )


def test_no_thread_leak_across_restarts(make_thread):
    rec = Recorder()
    t = make_thread(target=rec)
    t.Start()
    baseline = threading.active_count()

    for _ in range(10):
        assert wait_for(rec.advanced(1))
        t.Wait(DEADLINE)
        assert t.Restart() is True
    t.Wait(DEADLINE)
    assert wait_for(lambda: not is_alive(t))

    assert wait_for(lambda: threading.active_count() <= baseline), (
        f"thread count grew {baseline} -> {threading.active_count()}"
    )


# ---------------------------------------------------------------------------
# helpers that need to patch the module under test
# ---------------------------------------------------------------------------


class _silence_prints:
    """Shadow the builtin print inside thread.py with a no-op.

    A module global beats the builtin at lookup time, so this only affects the
    module under test.
    """

    def __enter__(self):
        thread_mod.print = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        del thread_mod.print
        return False


class _os_shim:
    """Swap thread.py's `os` global for a recorder that can inject read errors.

    Scoped to the module global so pytest's own use of os.read is untouched.
    """

    def __init__(self, read_error=None):
        self._read_error = read_error
        self._real = thread_mod.os
        self.opened = []
        self.closed = []

    def read(self, fd, n):
        if fd not in self.opened:
            self.opened.append(fd)
        if self._read_error is not None:
            raise self._read_error
        return self._real.read(fd, n)

    def close(self, fd):
        self.closed.append(fd)
        return self._real.close(fd)

    def __getattr__(self, item):
        return getattr(self._real, item)

    def __enter__(self):
        thread_mod.os = self
        return self

    def __exit__(self, *exc):
        thread_mod.os = self._real
        return False
