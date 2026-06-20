from __future__ import annotations

import threading
from contextlib import contextmanager


class CacheEntryLock:
    def __init__(self):
        """
        Per-entry locking mechanism for a shared write-once cache.

        Allows lockless reads (safe since entries are never mutated after insertion)
        while serializing concurrent writes to the same cache key via per-entry locks.

        Architecture
        ------------
        - `cache`       — the shared dict, read without any lock (write-once entries)
        - `_key_locks`  — one lock per key currently being written, cleaned up after write
        - `_meta_lock`  — short-lived lock only for creating/accessing `_key_locks` entries

        Usage
        -----
        ```python
        lock  = CacheEntryLock()
        cache = {}

        if key in cache:                   # lockless read
            use cache[key]
        else:
            with lock.for_key(key):        # per-entry lock, auto-cleaned up on exit
                if key in cache:           # double-check after acquiring
                    use cache[key]
                else:
                    cache[key] = compute() # only one thread reaches here per key
        ```

        Thread Safety
        -------------
        Safe under CPython's GIL — `dict` reads and writes are atomic per operation.
        Under free-threaded Python (3.13+ no-GIL), lockless reads would require an atomic dict or 
        memory barrier to remain correct.
        """

        self._meta_lock: threading.Lock            = threading.Lock()
        self._key_locks: dict[any, threading.Lock] = {}


    @contextmanager
    def for_key(self, key):
        """
        Context manager that acquires the per-entry lock for `key` and releases it — along with 
        removing it from `_key_locks` — on exit. Cleanup is automatic so callers cannot leak locks.

        `_meta_lock` is held only for the brief lookup/creation of the per-entry lock, not during 
        the actual computation or cache write.
        """

        with self._meta_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            key_lock = self._key_locks[key]

        with key_lock:
            yield

        with self._meta_lock:
            self._key_locks.pop(key, None)