"""Opt-in tests for the deployed Start/Wait thread lifecycle and timer cleanup.

Each controller loop owns one RecurrentThread instance. A replacement loop uses
a new instance; the current deployment API has no Restart operation. SDK imports
stay inside fixtures so portable test collection does not require the extra.
"""

import errno
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unitree

INTERVAL = 0.01  # Default control-loop period; the CSV writer uses no timer.
DEADLINE = 2.0


@pytest.fixture(scope="module")
def sdk_thread():
    if sys.platform != "linux":
        pytest.skip("Unitree timerfd thread tests require Linux")
    try:
        from unitree_sdk2py.utils.future import FutureResult

        from go2_deploy.utility import thread as thread_mod
    except ModuleNotFoundError as error:
        if error.name == "unitree_sdk2py" or error.name == "cyclonedds":
            pytest.fail(
                "Unitree tests requested but the SDK extra is missing; install "
                "it with `uv sync --frozen --extra unitree_sdk`.",
                pytrace=False,
            )
        raise
    return SimpleNamespace(module=thread_mod, future_result=FutureResult)


def wait_for(predicate):
    deadline = time.monotonic() + DEADLINE
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


class Recorder:
    def __init__(self, raise_error=False):
        self._lock = threading.Lock()
        self.stamps = []
        self.calls = []
        self.raise_error = raise_error

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.stamps.append(time.monotonic())
            self.calls.append((args, dict(kwargs)))
        if self.raise_error:
            raise RuntimeError("injected target failure")

    @property
    def count(self):
        with self._lock:
            return len(self.stamps)


@pytest.fixture
def timer_runtime(sdk_thread, monkeypatch):
    """Observe real Linux timers and inject errors only in the module under test."""
    module = sdk_thread.module
    runtime = SimpleNamespace(
        module=module,
        future_result=sdk_thread.future_result,
        created=[],
        closed=[],
        read_error=None,
    )
    original_create = module.timerfd_create

    def create(clock, flags):
        descriptor = original_create(clock, flags)
        runtime.created.append(descriptor)
        return descriptor

    def read(descriptor, length):
        if runtime.read_error is not None:
            error = runtime.read_error
            runtime.read_error = None
            raise error
        return os.read(descriptor, length)

    def close(descriptor):
        runtime.closed.append(descriptor)
        return os.close(descriptor)

    monkeypatch.setattr(module, "timerfd_create", create)
    monkeypatch.setattr(module, "os", SimpleNamespace(read=read, close=close))
    return runtime


@pytest.fixture
def make_thread(timer_runtime):
    created = []

    def make(target, *, interval=INTERVAL, name=None, args=(), kwargs=None):
        thread = timer_runtime.module.RecurrentThread(
            interval=interval, target=target, name=name, args=args, kwargs=kwargs
        )
        created.append(thread)
        return thread

    yield make

    # Teardown must remain reliable even if the tested Wait implementation fails.
    for thread in created:
        thread._RecurrentThread__quit = True
    for thread in created:
        native_thread = thread._Thread__thread
        if native_thread.ident is not None:
            native_thread.join(DEADLINE)
        assert not native_thread.is_alive(), "recurrent worker leaked after test"


def test_loop_runs_at_configured_period(make_thread):
    recorder = Recorder()
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 12)
    assert thread.Wait(DEADLINE) is True

    gaps = sorted(b - a for a, b in zip(recorder.stamps, recorder.stamps[1:]))
    median = gaps[len(gaps) // 2]
    assert INTERVAL * 0.5 < median < INTERVAL * 2.0, f"median period: {median}"


def test_wait_stops_target_calls(make_thread):
    recorder = Recorder()
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 3)
    assert thread.Wait(DEADLINE) is True
    assert wait_for(lambda: not thread.IsAlive())

    count = recorder.count
    time.sleep(INTERVAL * 3)
    assert recorder.count == count


def test_target_receives_args_and_kwargs(make_thread):
    recorder = Recorder()
    thread = make_thread(recorder, args=(1, 2), kwargs={"key": "value"})
    thread.Start()
    assert wait_for(lambda: recorder.count >= 3)
    assert thread.Wait(DEADLINE) is True
    assert all(call == ((1, 2), {"key": "value"}) for call in recorder.calls)


def test_zero_interval_repeats_blocking_target_and_stops_without_timer(
    make_thread, timer_runtime
):
    entered = threading.Event()
    release = threading.Semaphore(0)
    calls = []

    def target(value, *, key):
        calls.append((value, key))
        entered.set()
        # Like the CSV writer, the callback blocks until work is available.
        release.acquire(timeout=DEADLINE)

    thread = make_thread(target, interval=0.0, args=(7,), kwargs={"key": "value"})
    thread.Start()
    try:
        assert entered.wait(DEADLINE)
        entered.clear()
        release.release()
        assert entered.wait(DEADLINE)
        assert calls == [(7, "value"), (7, "value")]
        assert thread.Wait(0) is False
    finally:
        release.release()

    assert thread.Wait(DEADLINE) is True
    assert wait_for(lambda: not thread.IsAlive())
    assert calls == [(7, "value"), (7, "value")]
    assert timer_runtime.created == []


def test_thread_identity_is_available_while_running(make_thread):
    recorder = Recorder()
    thread = make_thread(recorder, name="go2-lowcmd-test")
    thread.Start()
    assert wait_for(lambda: recorder.count >= 1)
    assert thread.IsAlive()
    assert thread.GetId() == thread._Thread__thread.ident
    assert thread.GetNativeId() == thread._Thread__thread.native_id
    assert thread._Thread__thread.name == "go2-lowcmd-test"
    assert thread.Wait(DEADLINE) is True


def test_wait_timeout_reports_a_still_running_target(make_thread):
    entered = threading.Event()
    release = threading.Event()

    def target():
        entered.set()
        release.wait(DEADLINE)

    thread = make_thread(target)
    thread.Start()
    try:
        assert entered.wait(DEADLINE)
        assert thread.Wait(INTERVAL / 10) is False
        assert thread.IsAlive()
    finally:
        release.set()
    assert thread.Wait(DEADLINE) is True
    assert wait_for(lambda: not thread.IsAlive())


def test_get_result_times_out_without_stopping_loop(make_thread, timer_runtime):
    recorder = Recorder()
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 1)

    result = thread.GetResult(INTERVAL * 2)

    assert result.code == timer_runtime.future_result.FUTUTE_ERR_TIMEOUT
    assert thread.IsAlive()
    assert thread.Wait(DEADLINE) is True


def test_target_exception_does_not_stop_periodic_loop(make_thread):
    recorder = Recorder(raise_error=True)
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 4)
    assert thread.IsAlive()
    assert thread.Wait(DEADLINE) is True


def test_normal_stop_closes_timer(make_thread, timer_runtime):
    recorder = Recorder()
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 2)
    assert thread.Wait(DEADLINE) is True
    assert wait_for(lambda: not thread.IsAlive())
    assert len(timer_runtime.created) == 1
    assert timer_runtime.closed == timer_runtime.created


def test_timer_read_failure_sets_failed_result_and_closes_timer(
    make_thread, timer_runtime
):
    timer_runtime.read_error = OSError(errno.EIO, "injected read failure")
    thread = make_thread(Recorder())
    thread.Start()
    assert wait_for(lambda: not thread.IsAlive())
    assert (
        thread.GetResult(DEADLINE).code == timer_runtime.future_result.FUTURE_ERR_FAILED
    )
    assert len(timer_runtime.created) == 1
    assert timer_runtime.closed == timer_runtime.created


def test_timer_arm_failure_closes_created_descriptor(
    make_thread, timer_runtime, monkeypatch
):
    def fail_arm(*args):
        raise OSError(errno.EINVAL, "injected timer arm failure")

    monkeypatch.setattr(timer_runtime.module, "timerfd_settime", fail_arm)
    thread = make_thread(Recorder())
    thread.Start()
    assert wait_for(lambda: not thread.IsAlive())
    assert (
        thread.GetResult(DEADLINE).code == timer_runtime.future_result.FUTURE_ERR_FAILED
    )
    assert len(timer_runtime.created) == 1
    assert timer_runtime.closed == timer_runtime.created


def test_temporary_read_unavailability_keeps_loop_running(make_thread, timer_runtime):
    timer_runtime.read_error = OSError(errno.EAGAIN, "injected temporary failure")
    recorder = Recorder()
    thread = make_thread(recorder)
    thread.Start()
    assert wait_for(lambda: recorder.count >= 4)
    assert thread.IsAlive()
    assert thread.Wait(DEADLINE) is True
    assert wait_for(lambda: not thread.IsAlive())
    assert timer_runtime.closed == timer_runtime.created


def test_replacement_instances_do_not_leave_live_workers_or_timers(
    make_thread, timer_runtime
):
    for _ in range(5):
        recorder = Recorder()
        thread = make_thread(recorder)
        thread.Start()
        assert wait_for(lambda: recorder.count >= 2)
        assert thread.Wait(DEADLINE) is True
        assert wait_for(lambda: not thread.IsAlive())
        assert timer_runtime.closed == timer_runtime.created
    assert len(timer_runtime.created) == 5
