import threading
import time

from app.concurrency import serialize_first_call


def _slow_builder(state):
    @serialize_first_call
    def build():
        with state["lock"]:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with state["lock"]:
            state["active"] -= 1

    return build


def test_two_different_getters_never_build_at_the_same_time():
    """One shared lock, not one per getter: transformers' lazy imports are not thread-safe, and the
    first scene_description request after a restart (grounding scan + captioner loading in parallel)
    lost an import race live."""
    state = {"lock": threading.Lock(), "active": 0, "peak": 0}
    threads = [threading.Thread(target=_slow_builder(state)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["peak"] == 1


def test_a_getter_that_calls_another_getter_does_not_deadlock():
    @serialize_first_call
    def inner():
        return "inner"

    @serialize_first_call
    def outer():
        return inner() + "+outer"

    result = []
    worker = threading.Thread(target=lambda: result.append(outer()))
    worker.start()
    worker.join(timeout=5)
    assert result == ["inner+outer"]
