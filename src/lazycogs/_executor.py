"""Shared thread pool executor for CPU-bound reprojection work."""

from __future__ import annotations

import concurrent.futures
import os

_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_MAX_WORKERS: int | None = None


def _default_workers() -> int:
    """Return the default worker count: up to 4 per available CPU, capped at 4.

    Conservative by default so lazycogs does not peg all cores on a shared
    JupyterHub.  Call :func:`set_reproject_workers` to raise the limit on
    dedicated hardware.
    """
    return min(os.cpu_count() or 4, 4)


def get_reproject_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared reprojection thread pool, creating it on first call.

    Returns:
        A ``ThreadPoolExecutor`` bounded to the worker count set by
        :func:`set_reproject_workers` (default: ``min(os.cpu_count(), 4)``).

    """
    global _EXECUTOR, _MAX_WORKERS
    if _EXECUTOR is None:
        if _MAX_WORKERS is None:
            _MAX_WORKERS = _default_workers()
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="lazycogs-reproject",
        )
    return _EXECUTOR


def set_reproject_workers(n: int) -> None:
    """Set the number of threads used for CPU-bound reprojection work.

    Call this before any reads to tune for your environment.  The default is
    ``min(os.cpu_count(), 4)``, which is conservative for shared environments
    like JupyterHub.  On a dedicated machine you can raise it toward
    ``os.cpu_count()``.

    Because this executor is shared across all ``lazycogs.open()`` calls in the
    process, a single bounded pool is used for reprojection regardless of how
    many dask workers are running concurrently.  This prevents thread
    proliferation under heavy dask parallelism.

    Args:
        n: Number of worker threads.  Must be >= 1.

    Raises:
        ValueError: If ``n`` is less than 1.

    """
    global _EXECUTOR, _MAX_WORKERS
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    _MAX_WORKERS = n
    _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
        max_workers=n,
        thread_name_prefix="lazycogs-reproject",
    )
