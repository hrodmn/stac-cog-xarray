"""Tests for _store.resolve."""

import pytest
from obstore.store import MemoryStore

from lazycogs._store import resolve


@pytest.mark.parametrize(
    "href, expected_path",
    [
        ("s3://my-bucket/path/to/file.tif", "path/to/file.tif"),
        ("s3a://my-bucket/deep/path/file.tif", "deep/path/file.tif"),
        ("gs://my-bucket/data/image.tif", "data/image.tif"),
        ("https://example.com/data/image.tif", "data/image.tif"),
        ("http://localhost:8080/tiles/tile.tif", "tiles/tile.tif"),
    ],
)
def test_path_extraction(href, expected_path):
    """The returned path is the URL path component without a leading slash."""
    _, path = resolve(href)
    assert path == expected_path


@pytest.mark.parametrize(
    "href",
    [
        "s3://bucket/a.tif",
        "s3a://bucket/a.tif",
        "gs://bucket/a.tif",
        "https://host.com/a.tif",
        "http://host.com/a.tif",
    ],
)
def test_store_is_not_none(href):
    """A store object is returned for all supported schemes."""
    store, _ = resolve(href)
    assert store is not None


def test_unsupported_scheme_raises():
    """obstore's from_url rejects unknown schemes."""
    with pytest.raises(Exception, match="(?i)scheme|url"):
        resolve("ftp://server/file.tif")


def test_thread_local_cache_same_bucket():
    """Two HREFs in the same bucket return the same store object."""
    store_a, _ = resolve("s3://shared-bucket/file1.tif")
    store_b, _ = resolve("s3://shared-bucket/file2.tif")
    assert store_a is store_b


def test_thread_local_cache_different_buckets():
    """HREFs in different buckets return distinct store objects."""
    store_a, _ = resolve("s3://bucket-one/file.tif")
    store_b, _ = resolve("s3://bucket-two/file.tif")
    assert store_a is not store_b


def test_thread_local_cache_same_https_host():
    """Two HTTPS HREFs on the same host share a store."""
    store_a, _ = resolve("https://cdn.example.com/img/a.tif")
    store_b, _ = resolve("https://cdn.example.com/img/b.tif")
    assert store_a is store_b


def test_user_supplied_store_is_returned_unchanged():
    """When a store is passed, it is returned as-is with just the path extracted."""
    user_store = MemoryStore()
    store, path = resolve("s3://bucket/some/key.tif", store=user_store)
    assert store is user_store
    assert path == "some/key.tif"


def test_user_supplied_store_bypasses_cache():
    """Passing a store should never consult or populate the auto-cache."""
    user_store = MemoryStore()
    store, _ = resolve("s3://never-cached-bucket/a.tif", store=user_store)
    assert store is user_store
    auto_store, _ = resolve("s3://never-cached-bucket/b.tif")
    assert auto_store is not user_store
