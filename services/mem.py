"""
services/mem.py — aggressive memory release helpers for Render's 512 MB tier.

Python frees objects but the underlying glibc allocator usually KEEPS that
memory reserved for the process (it doesn't hand it back to the OS). On a
512 MB box this makes RAM ratchet upward request after request until a spike
triggers an OOM kill.

`release_memory()` runs a full GC and then calls glibc `malloc_trim(0)` to
actually return the freed arenas to the OS. Call it after every heavy
operation (uploads, embedding, reranking, chat).
"""

import ctypes
import ctypes.util
import gc

_libc = None
_have_trim = False

try:
    _name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_name)
    _have_trim = hasattr(_libc, "malloc_trim")
except Exception:
    _libc = None
    _have_trim = False


def release_memory():
    """Full GC + return freed heap back to the OS (best-effort)."""
    try:
        gc.collect()
    except Exception:
        pass
    if _have_trim:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def mem_usage_mb():
    """Current resident set size in MB (best-effort, Linux only)."""
    try:
        import os
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return -1.0
