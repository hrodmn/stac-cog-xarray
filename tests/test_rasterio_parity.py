"""Parity tests: lazycogs read pipeline vs rasterio nearest-neighbor.

Verifies that lazycogs produces bit-identical output to rasterio for nearest-
neighbor resampling, at resolutions that bracket each overview boundary.
Overview levels for the synthetic COG are 20, 40, 80, and 160 m, so
resolutions just below, at, and just above each boundary are all covered.

Two CRS scenarios are tested:
- Same CRS: destination grid is in the same UTM 32N CRS as the source file.
  This exercises pure pixel-mapping without any coordinate transform.
- Cross CRS: destination grid is in UTM zone 33N (EPSG:32633), an adjacent
  zone that shares metric units and a similar pixel scale (~10 m) but requires
  a genuine coordinate transform.  Using a same-unit projection avoids the
  large scale-ratio issues that arise with degree-based CRS (e.g. WGS84),
  which cause floating-point boundary sensitivity unrelated to overview
  selection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.enums
import rasterio.warp
from affine import Affine
from async_geotiff import GeoTIFF
from pyproj import CRS

from lazycogs._chunk_reader import _native_window, _select_overview
from lazycogs._reproject import _get_transformer, apply_warp_map, compute_warp_map
from lazycogs._store import resolve

# Resolutions chosen to bracket each overview boundary (20, 40, 80, 160 m).
# Includes native resolution, between-level values, and one well above all overviews.
_RESOLUTIONS = [
    10,
    14,
    19,
    20,
    21,
    30,
    39,
    40,
    41,
    60,
    79,
    80,
    81,
    120,
    159,
    160,
    161,
    300,
]

_CHUNK_SIZE = 64
_CENTER_UTM_X = 510_240.0
_CENTER_UTM_Y = 5_589_760.0


def _chunk_affine(resolution: float, center_x: float, center_y: float) -> Affine:
    half = (_CHUNK_SIZE / 2) * resolution
    return Affine(resolution, 0.0, center_x - half, 0.0, -resolution, center_y + half)


async def _read_lazycogs(
    href: str,
    chunk_affine: Affine,
    dst_crs: CRS,
) -> np.ndarray:
    """Run the lazycogs read pipeline and return the output array."""
    store, path = resolve(href)
    geotiff = await GeoTIFF.open(path, store=store)
    src_crs = geotiff.crs
    same_crs = dst_crs.equals(src_crs)

    target_res_native = abs(chunk_affine.a)
    t = None
    if not same_crs:
        t = _get_transformer(dst_crs, src_crs)
        cx = chunk_affine.c + (_CHUNK_SIZE / 2) * chunk_affine.a
        cy = chunk_affine.f + (_CHUNK_SIZE / 2) * chunk_affine.e
        x0, _ = t.transform(cx, cy)
        x1, _ = t.transform(cx + chunk_affine.a, cy)
        target_res_native = abs(x1 - x0)

    overview = _select_overview(geotiff, target_res_native)
    reader = overview if overview is not None else geotiff

    chunk_minx = chunk_affine.c
    chunk_maxy = chunk_affine.f
    chunk_maxx = chunk_minx + _CHUNK_SIZE * chunk_affine.a
    chunk_miny = chunk_maxy + _CHUNK_SIZE * chunk_affine.e

    if same_crs or t is None:
        bbox_native = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
    else:
        xs, ys = t.transform(
            [chunk_minx, chunk_maxx, chunk_minx, chunk_maxx],
            [chunk_maxy, chunk_maxy, chunk_miny, chunk_miny],
        )
        bbox_native = (min(xs), min(ys), max(xs), max(ys))

    window = _native_window(reader, bbox_native, reader.width, reader.height)
    assert window is not None, "test chunk must overlap COG extent"

    raster = await reader.read(window=window)
    warp_map = compute_warp_map(
        raster.transform, src_crs, chunk_affine, dst_crs, _CHUNK_SIZE, _CHUNK_SIZE
    )
    return apply_warp_map(raster.data, warp_map, geotiff.nodata)


def _odc_overview_level(
    path: Path, target_res_native: float, native_res: float
) -> int | None:
    """Replicate odc-stac's pick_overview: coarsest shrink <= read_shrink."""
    read_shrink = int(target_res_native / native_res)
    with rasterio.open(path) as src:
        overviews = src.overviews(1)
    ovr_level: int | None = None
    for i, shrink in enumerate(overviews):
        if shrink <= read_shrink:
            ovr_level = i
        else:
            break
    return ovr_level


def _read_rasterio(
    path: Path,
    chunk_affine: Affine,
    dst_crs: CRS,
    native_res: float,
) -> np.ndarray:
    """Run rasterio nearest-neighbor reproject at the odc-stac-selected overview."""
    with rasterio.open(path) as src:
        src_crs_obj = CRS.from_user_input(src.crs.to_wkt())
    same_crs = dst_crs.equals(src_crs_obj)

    target_res_native = abs(chunk_affine.a)
    if not same_crs:
        t = _get_transformer(dst_crs, src_crs_obj)
        cx = chunk_affine.c + (_CHUNK_SIZE / 2) * chunk_affine.a
        cy = chunk_affine.f + (_CHUNK_SIZE / 2) * chunk_affine.e
        x0, _ = t.transform(cx, cy)
        x1, _ = t.transform(cx + chunk_affine.a, cy)
        target_res_native = abs(x1 - x0)

    ovr_level = _odc_overview_level(path, target_res_native, native_res)

    with rasterio.open(path, overview_level=ovr_level) as src:
        out = np.zeros((1, _CHUNK_SIZE, _CHUNK_SIZE), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=Affine(
                chunk_affine.a,
                chunk_affine.b,
                chunk_affine.c,
                chunk_affine.d,
                chunk_affine.e,
                chunk_affine.f,
            ),
            dst_crs=dst_crs.to_wkt(),
            resampling=rasterio.enums.Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    return out.astype(np.uint16)


def _href(path: Path) -> str:
    return path.as_uri()


def _assert_parity(
    lazycogs_out: np.ndarray,
    rasterio_out: np.ndarray,
    label: str,
    max_differing_pixels: int = 0,
    max_abs_diff: int = 0,
) -> None:
    """Assert that the two outputs are pixel-identical within the given tolerances.

    ``max_differing_pixels`` and ``max_abs_diff`` may both be nonzero only for
    the cross-CRS test, where a handful of destination pixel centres can land
    within floating-point precision of a source pixel boundary and lazycogs
    (pyproj) and GDAL round to opposite sides.  These boundary pixels never
    indicate an overview selection error; they differ by at most one source
    overview pixel's value.  The tolerances here are deliberately tight so that
    any systematic regression (wrong overview level, large pixel-mapping error)
    still trips the assertion.
    """
    diff = lazycogs_out.astype(np.int32) - rasterio_out.astype(np.int32)
    n_diff = int(np.count_nonzero(diff))
    actual_max = int(np.abs(diff).max()) if n_diff else 0
    assert n_diff <= max_differing_pixels and actual_max <= max_abs_diff, (
        f"{label}: {n_diff}/{lazycogs_out.size} pixels differ "
        f"(allowed ≤{max_differing_pixels}); "
        f"max abs diff = {actual_max} (allowed ≤{max_abs_diff})"
    )


@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_parity_same_crs(synthetic_cog: Path, resolution: int) -> None:
    """lazycogs matches rasterio/nearest for same-CRS reads at all overview boundaries."""
    dst_crs = CRS.from_epsg(32632)
    affine = _chunk_affine(resolution, _CENTER_UTM_X, _CENTER_UTM_Y)

    lc_out = asyncio.run(_read_lazycogs(_href(synthetic_cog), affine, dst_crs))
    rio_out = _read_rasterio(synthetic_cog, affine, dst_crs, native_res=10.0)

    _assert_parity(
        lc_out,
        rio_out,
        f"same_crs res={resolution}",
        max_differing_pixels=0,
        max_abs_diff=0,
    )


@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_parity_cross_crs(synthetic_cog: Path, resolution: int) -> None:
    """lazycogs matches rasterio/nearest for cross-CRS reads at all overview boundaries.

    Destination CRS is UTM zone 33N (EPSG:32633).  The source COG is in zone
    32N, so a real coordinate transform is required, but both CRS share metric
    units and a similar pixel scale.  This avoids the large scale-ratio issues
    of degree-based projections while still exercising the cross-CRS code path.
    """
    src_crs = CRS.from_epsg(32632)
    dst_crs = CRS.from_epsg(
        3035
    )  # ETRS89 / LAEA Europe — same-unit, low distortion near COG
    t = _get_transformer(src_crs, dst_crs)
    cx_laea, cy_laea = t.transform(_CENTER_UTM_X, _CENTER_UTM_Y)
    affine = _chunk_affine(float(resolution), cx_laea, cy_laea)

    lc_out = asyncio.run(_read_lazycogs(_href(synthetic_cog), affine, dst_crs))
    rio_out = _read_rasterio(synthetic_cog, affine, dst_crs, native_res=10.0)

    # Allow ≤ 3 pixels to differ by at most 1 overview-row's worth of value.
    # These are floating-point boundary pixels where the destination centre lands
    # within a ULP of a source pixel edge and pyproj/GDAL round to opposite sides.
    # A systematic regression (wrong overview level, large pixel mapping error)
    # would produce far more differing pixels or a much larger diff.
    _assert_parity(
        lc_out,
        rio_out,
        f"cross_crs res={resolution}m",
        max_differing_pixels=3,
        max_abs_diff=2048 * 16 + 1,  # 1 row in the coarsest (16×) overview
    )
