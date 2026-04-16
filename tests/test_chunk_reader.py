"""Tests for _chunk_reader helpers: _select_overview, _native_window, and async_mosaic_chunk."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
from affine import Affine
from pyproj import CRS

from stac_cog_xarray._chunk_reader import (
    _native_window,
    _select_overview,
    async_mosaic_chunk,
)
from stac_cog_xarray._mosaic_methods import FirstMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_geotiff(native_res: float, overview_resolutions: list[float]) -> MagicMock:
    """Build a minimal GeoTIFF mock with the given resolutions."""
    geotiff = MagicMock()
    geotiff.transform = Affine(native_res, 0.0, 0.0, 0.0, -native_res, 0.0)

    overviews = []
    for res in overview_resolutions:
        ov = MagicMock()
        ov.transform = Affine(res, 0.0, 0.0, 0.0, -res, 0.0)
        overviews.append(ov)

    geotiff.overviews = overviews
    return geotiff


def _mock_reader(transform: Affine) -> MagicMock:
    """Build a minimal GeoTIFF/Overview mock with the given transform."""
    reader = MagicMock()
    reader.transform = transform
    return reader


# ---------------------------------------------------------------------------
# _select_overview
# ---------------------------------------------------------------------------


def test_select_overview_no_overviews_returns_none():
    """Returns None when the file has no overviews."""
    geotiff = _mock_geotiff(10.0, [])
    assert _select_overview(geotiff, 100.0) is None


def test_select_overview_target_finer_than_native_returns_none():
    """Returns None when the requested resolution is finer than native."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    assert _select_overview(geotiff, 5.0) is None


def test_select_overview_target_equal_to_native_returns_none():
    """Returns None when the requested resolution equals native."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    assert _select_overview(geotiff, 10.0) is None


def test_select_overview_returns_finest_sufficient_overview():
    """Returns the finest overview whose resolution >= target."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0, 80.0])
    ov = _select_overview(geotiff, 30.0)
    # Target is 30 m → finest overview >= 30 m is the 40 m one (index 1)
    assert ov is geotiff.overviews[1]


def test_select_overview_exact_match():
    """Returns the overview whose resolution exactly matches the target."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0])
    ov = _select_overview(geotiff, 20.0)
    assert ov is geotiff.overviews[0]


def test_select_overview_target_coarser_than_all_overviews():
    """When target is coarser than all overviews, returns the coarsest."""
    geotiff = _mock_geotiff(10.0, [20.0, 40.0, 80.0])
    ov = _select_overview(geotiff, 200.0)
    assert ov is geotiff.overviews[-1]


# ---------------------------------------------------------------------------
# _native_window
# ---------------------------------------------------------------------------


def test_native_window_full_coverage():
    """A bbox that covers the full image returns a window matching the image."""
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)  # 4-px tall, any width
    reader = _mock_reader(transform)
    win = _native_window(reader, (0.0, 0.0, 4.0, 4.0), width=4, height=4)
    assert win is not None
    assert win.col_off == 0
    assert win.row_off == 0
    assert win.width == 4
    assert win.height == 4


def test_native_window_sub_region():
    """A bbox covering the bottom-right quadrant returns the correct window."""
    # 8×8 image, 1 m resolution, origin top-left at (0, 8)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0)
    reader = _mock_reader(transform)
    # Bottom-right quadrant: x=[4,8], y=[0,4]
    win = _native_window(reader, (4.0, 0.0, 8.0, 4.0), width=8, height=8)
    assert win is not None
    assert win.col_off == 4
    assert win.row_off == 4
    assert win.width == 4
    assert win.height == 4


def test_native_window_bbox_outside_returns_none():
    """A bbox entirely outside the image returns None."""
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    reader = _mock_reader(transform)
    # Image covers x=[0,4], bbox is at x=[10,14]
    win = _native_window(reader, (10.0, 0.0, 14.0, 4.0), width=4, height=4)
    assert win is None


def test_native_window_clamped_to_image_bounds():
    """A bbox that extends beyond image edges is clamped to valid pixels."""
    # 4×4 image
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    reader = _mock_reader(transform)
    # Bbox extends 2 pixels beyond the right and bottom edges
    win = _native_window(reader, (2.0, -2.0, 6.0, 2.0), width=4, height=4)
    assert win is not None
    assert win.col_off == 2
    assert win.col_off + win.width <= 4  # clamped at image width
    assert win.row_off + win.height <= 4  # clamped at image height


# ---------------------------------------------------------------------------
# async_mosaic_chunk concurrency
# ---------------------------------------------------------------------------


def test_async_mosaic_chunk_limits_concurrent_reads():
    """No more than max_concurrent_reads items are read concurrently."""
    chunk_width, chunk_height = 4, 4
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    dst_crs = CRS.from_epsg(4326)
    n_items = 20
    max_concurrent = 5

    peak_concurrent = [0]
    current_concurrent = [0]

    async def _fake_read_item_band(*args, **kwargs):
        current_concurrent[0] += 1
        peak_concurrent[0] = max(peak_concurrent[0], current_concurrent[0])
        await asyncio.sleep(0)  # yield to allow other coroutines to increment
        arr = np.ones((1, chunk_height, chunk_width), dtype=np.float32)
        current_concurrent[0] -= 1
        return arr, None

    items = [{"id": f"item-{i}", "assets": {}} for i in range(n_items)]

    with patch(
        "stac_cog_xarray._chunk_reader._read_item_band",
        side_effect=_fake_read_item_band,
    ):
        asyncio.run(
            async_mosaic_chunk(
                items=items,
                band="B01",
                chunk_affine=chunk_affine,
                dst_crs=dst_crs,
                chunk_width=chunk_width,
                chunk_height=chunk_height,
                nodata=None,
                max_concurrent_reads=max_concurrent,
            )
        )

    assert peak_concurrent[0] <= max_concurrent


def test_async_mosaic_chunk_early_exit_skips_remaining_reads():
    """FirstMethod stops fetching items once all pixels are filled."""
    chunk_width, chunk_height = 4, 4
    chunk_affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    dst_crs = CRS.from_epsg(4326)
    n_items = 30
    # Each call returns a fully valid array, so FirstMethod completes on the
    # first item.  With batch size 5, only the first batch (5 reads) should
    # execute — 25 items should be skipped.
    reads_executed = [0]

    async def _fake_read_item_band(*args, **kwargs):
        reads_executed[0] += 1
        arr = np.ones((1, chunk_height, chunk_width), dtype=np.float32)
        return arr, None

    items = [{"id": f"item-{i}", "assets": {}} for i in range(n_items)]

    with patch(
        "stac_cog_xarray._chunk_reader._read_item_band",
        side_effect=_fake_read_item_band,
    ):
        asyncio.run(
            async_mosaic_chunk(
                items=items,
                band="B01",
                chunk_affine=chunk_affine,
                dst_crs=dst_crs,
                chunk_width=chunk_width,
                chunk_height=chunk_height,
                nodata=None,
                mosaic_method=FirstMethod(),
                max_concurrent_reads=5,
            )
        )

    # Only one batch of 5 should have been read; the mosaic fills on item 0
    # but the whole batch still runs.  Critically, the remaining 25 items
    # should NOT be read.
    assert reads_executed[0] <= 5
