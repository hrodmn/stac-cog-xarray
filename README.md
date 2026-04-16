# stac-cog-xarray

Open a geoparquet STAC item collection as a lazy `(time, band, y, x)` xarray DataArray backed by Cloud Optimized GeoTIFFs. No GDAL required.

## Why

Most tools that combine STAC and xarray (stackstac, odc-stac, rioxarray's GTI backend) depend on GDAL for spatial indexing, COG I/O, and reprojection. GDAL works, but it introduces a large build-time dependency that is difficult to distribute as a standard wheel.

This package replaces GDAL with a set of modern, Rust-backed libraries that ship as standard Python wheels:

| Task | Library |
|---|---|
| STAC search + spatial indexing | `rustac` (DuckDB + geoparquet) |
| COG I/O | `async-geotiff` (Rust, no GDAL) |
| Cloud storage | `obstore` |
| Reprojection | `pyproj` + numpy |
| Lazy dataset construction | xarray `BackendEntrypoint` + `LazilyIndexedArray` |

## Installation

Not yet published to PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/hrodmn/stac-cog-xarray.git
```

## Quickstart

```python
import rustac
import stac_cog_xarray

# Search a STAC API and write results to a local geoparquet file
rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

# Open the parquet file as a lazy (time, band, y, x) DataArray
da = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
)
# No COGs have been read yet

# Use time_period="P1W" to composite items within each ISO calendar week.
# The default FirstMethod fills each pixel from the first item with a valid
# (non-nodata) value, skipping remaining items in the week once all pixels
# are filled. This is more efficient than post-hoc ffill or reductions over
# a daily array, which would materialise every time step before reducing.
da_weekly = stac_cog_xarray.open(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    time_period="P1W",
)
```

## Documentation

- [Demo notebook](https://hrodmn.github.io/stac-cog-xarray/demo/)
- [Architecture](https://hrodmn.github.io/stac-cog-xarray/architecture/)
- [API Reference](https://hrodmn.github.io/stac-cog-xarray/api/)
