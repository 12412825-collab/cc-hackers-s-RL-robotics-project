"""Thread-safe commands shared by the dashboard and simulator adapter."""

from threading import Event


_reset_requested = Event()
_reset_completed = Event()


def request_reset():
    _reset_completed.clear()
    _reset_requested.set()


def consume_reset():
    if not _reset_requested.is_set():
        return False
    _reset_requested.clear()
    return True


def acknowledge_reset():
    _reset_completed.set()


def wait_for_reset(timeout=2.0):
    return _reset_completed.wait(timeout)
