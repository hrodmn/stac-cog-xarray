![lazycogs](./lazycogs.svg)

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
pip install git+https://github.com/hrodmn/lazycogs.git
```

## Quickstart

```python
import rustac
import lazycogs

# Search a STAC API and write results to a local geoparquet file
await rustac.search_to(
    "items.parquet",
    "https://earth-search.aws.element84.com/v1",
    collections=["sentinel-2-l2a"],
    datetime="2023-06-01/2023-08-31",
    bbox=[-93.5, 44.5, -93.0, 45.0],
)

# Open the parquet file as a lazy (time, band, y, x) DataArray
da = await lazycogs.open_async(
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
da_weekly = await lazycogs.open_async(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    time_period="P1W",
)
```

## Inspecting read plans

Before computing an array, you can ask what DuckDB queries and COG reads would
fire without touching any pixel data. The `da.lazycogs.explain()` method runs
the same spatial queries as `.compute()` but stops before any I/O:

```python
da = await lazycogs.open_async(
    "items.parquet",
    bbox=(380000.0, 4928000.0, 420000.0, 4984000.0),
    crs="EPSG:32615",
    resolution=10.0,
    chunks={"time": 1, "x": 512, "y": 512},
)

# Inspect without reading pixels
plan = da.lazycogs.explain()
print(plan.summary())

# Explain a specific slice
plan_subset = da.isel(time=0).lazycogs.explain()

# Convert to a DataFrame for analysis
df = plan.to_dataframe()
df.groupby("band")["n_cog_reads"].describe()

# Fetch COG headers to see which overview level and pixel window would be read
plan_full = da.lazycogs.explain(fetch_headers=True)
print(plan_full.summary())  # shows overview level distribution and avg window size

# Inspect per-item overview and window details
df_full = plan_full.to_dataframe()
df_full[["item_id", "band", "overview_level", "overview_resolution", "window_width", "window_height"]]
```

The `ExplainPlan` returned shows how many items are matched per chunk, the
distribution of items-per-chunk (useful for spotting over-lapping scene edges),
and the empty-chunk fraction (useful for diagnosing sparse time series).

## Custom object stores

By default, `lazycogs.open()` parses each asset HREF into an obstore `ObjectStore` using [`obstore.store.from_url`](https://developmentseed.org/obstore/latest/api/store/from_url/). Native cloud schemes (`s3://`, `s3a://`, `gs://`) default to unsigned requests so public buckets work without credentials.

For anything else — authenticated buckets, signed URLs, request-payer buckets, custom endpoints, MinIO, Cloudflare R2 with an API token, etc. — construct the store yourself and pass it via `store=`. Only the path portion of each HREF is then used to locate objects; the store must be rooted at the same `scheme://netloc` the HREFs point to.

```python
from obstore.store import S3Store, GCSStore, HTTPStore

# Authenticated S3 (credentials from env or boto3 chain)
store = S3Store(bucket="my-private-bucket", region="us-west-2")
da = lazycogs.open("items.parquet", ..., store=store)

# Requester-pays S3
store = S3Store(bucket="usgs-landsat", region="us-west-2", request_payer=True)

# Signed HTTPS (e.g. a SAS-token URL issued by a STAC API)
store = HTTPStore.from_url("https://myaccount.blob.core.windows.net/container?sv=...")

# GCS with a service-account key
store = GCSStore(bucket="my-bucket", service_account_path="/path/to/key.json")
```

See the [obstore store docs](https://developmentseed.org/obstore/latest/api/store/) for the full set of constructors and options.

## Tuning concurrency

lazycogs uses two independent concurrency controls:

**`max_concurrent_reads`** (passed to `open()`, default 32) limits how many COG files are opened and read simultaneously within a single chunk. This is pure async I/O — it does not create threads. Lower it if you want to reduce peak memory per chunk or are hitting S3 request-rate limits.

**`set_reproject_workers`** controls how many threads each chunk's event loop uses for CPU-bound reprojection (pyproj + numpy). The default is `min(os.cpu_count(), 4)`. Reprojection is memory-bandwidth-bound rather than compute-bound — benchmarks show diminishing returns above 4 threads because concurrent large-array operations saturate the memory bus rather than adding throughput. Raising this beyond 4 is rarely useful.

Each chunk gets its own independent thread pool (not a shared global pool), so dask tasks do not queue behind each other for reprojection.

When using dask, total concurrent COG reads across all workers equals `dask_workers × max_concurrent_reads`. On a 16-core machine with default dask worker count (16) and `max_concurrent_reads=32`, that is 512 simultaneous reads. If you hit S3 throttling or memory pressure, reduce `max_concurrent_reads` at `open()` time.

For better throughput, add time parallelism via dask rather than raising reprojection workers:

```python
# parallelize across time steps — each step gets its own full event loop + thread pool
da = lazycogs.open("items.parquet", ..., chunks={"time": 1})
da.compute()
```

## Documentation

- [Demo notebook](https://hrodmn.github.io/lazycogs/demo/)
- [Architecture](https://hrodmn.github.io/lazycogs/architecture/)
- [API Reference](https://hrodmn.github.io/lazycogs/api/)
