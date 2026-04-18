"""Resolve cloud storage HREFs into obstore ``ObjectStore`` instances."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from obstore.store import from_url

if TYPE_CHECKING:
    from obstore.store import ObjectStore

_local = threading.local()

# Native cloud schemes where public-bucket access is the common default.
# HTTPS URLs are excluded: `from_url` routes known hosts (amazonaws.com, etc.)
# to the right store, and unsigned requests on a private bucket should fail
# loudly rather than be forced through here.
_PUBLIC_DEFAULT_SCHEMES = frozenset({"s3", "s3a", "gs"})


def _cache() -> dict[str, ObjectStore]:
    """Return the thread-local store cache, creating it on first access."""
    if not hasattr(_local, "stores"):
        _local.stores = {}
    return _local.stores


def resolve(href: str, store: ObjectStore | None = None) -> tuple[ObjectStore, str]:
    """Resolve an HREF into an ``(ObjectStore, path)`` pair.

    When ``store`` is supplied, it is returned unchanged and only the object
    path is extracted from the HREF. The caller is responsible for ensuring
    the store is rooted at the same ``scheme://netloc`` the HREF points to;
    no introspection is performed on the provided store.

    When ``store`` is ``None``, a store is auto-constructed via
    :func:`obstore.store.from_url` using only the ``scheme://netloc`` portion
    of the HREF and cached per thread. Native cloud schemes (``s3``, ``s3a``,
    ``gs``) default to ``skip_signature=True`` so public buckets work without
    credentials. For authenticated access, signed URLs, custom endpoints, or
    request-payer buckets, construct the store yourself and pass it via
    ``store`` — see the README for examples.

    Args:
        href: A storage URL supported by :func:`obstore.store.from_url`
            (``s3``, ``s3a``, ``gs``, Azure variants, ``http``, ``https``,
            ``file``, ``memory``).
        store: Optional pre-configured ``ObjectStore`` to use directly.

    Returns:
        A ``(store, path)`` tuple where ``path`` is the object path within
        the store (no leading slash, except for ``file://`` which keeps the
        absolute path).

    """
    parsed = urlparse(href)
    scheme = parsed.scheme.lower()
    path = parsed.path if scheme == "file" else parsed.path.lstrip("/")

    if store is not None:
        return store, path

    root_url = f"{scheme}://{parsed.netloc}" if scheme != "file" else "file:///"
    cache = _cache()
    if root_url not in cache:
        kwargs = {"skip_signature": True} if scheme in _PUBLIC_DEFAULT_SCHEMES else {}
        cache[root_url] = from_url(root_url, **kwargs)
    return cache[root_url], path
