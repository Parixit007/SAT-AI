"""Tiny shared thread-safety helper. Every specialist adapter's lazy-singleton getter (`if _tool is
None: build()` in grounding_adapter.py / water_segmentation_adapter.py / vqa_adapter.py) and
gee/client.py's `ensure_initialized()` use the same unguarded check-then-act pattern -- racy under
Starlette's `run_in_threadpool` (routes_query.py runs `handle_query` there specifically because
specialist calls are blocking), since two concurrent first requests can both start building the
same multi-hundred-MB-to-multi-GB model at once. Real risk, not theoretical: CLAUDE.md already
notes three resident models is tight on a 16GB machine.

Lives here (rather than under specialists/) so both specialists/*_adapter.py and gee/client.py can
import it without either package depending on the other."""

import functools
import threading


def serialize_first_call(fn):
    """Wrap a lazy-singleton getter/initializer so concurrent callers serialize instead of racing.
    Does NOT change where the cached value lives -- each wrapped function keeps its own
    module-level `global _x`, so existing tests that monkeypatch that global directly keep working
    unmodified. Locks on every call, not just the first, which is fine here: an uncontended lock
    acquisition is ~100ns, negligible next to the model-loading/inference this guards."""
    lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with lock:
            return fn(*args, **kwargs)

    return wrapper
