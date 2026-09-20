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

# ONE lock for every lazy model getter in the process, not one per getter. Two different models
# being built at the same moment is not just a memory spike: transformers' lazy submodule imports
# are not thread-safe, and the first scene_description request after a restart -- which builds the
# grounding detector and the captioner in parallel threads -- failed live with "Could not import
# module 'AutoProcessor'" (the captioner lost the race). Model construction happens once per model
# per process, so serialising it costs nothing that matters; inference is never inside this lock.
# Re-entrant so a getter that itself calls another getter can't deadlock against itself.
_MODEL_BUILD_LOCK = threading.RLock()


def serialize_first_call(fn):
    """Wrap a lazy-singleton getter/initializer so concurrent callers serialize instead of racing.
    Does NOT change where the cached value lives -- each wrapped function keeps its own
    module-level `global _x`, so existing tests that monkeypatch that global directly keep working
    unmodified. Locks on every call, not just the first, which is fine here: an uncontended lock
    acquisition is ~100ns, negligible next to the model-loading/inference this guards."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _MODEL_BUILD_LOCK:
            return fn(*args, **kwargs)

    return wrapper
