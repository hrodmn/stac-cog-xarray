"""End-to-end benchmarks using the public lazycogs.open() API.

These benchmarks require local benchmark data. See scripts/prepare_benchmark_data.py.

Run with:
    uv run pytest tests/benchmarks/ --benchmark-enable
    uv run pytest tests/benchmarks/ --benchmark-enable --benchmark-save=<name>
"""

import pytest

import lazycogs
from lazycogs import FirstMethod, MedianMethod, MosaicMethodBase

from .conftest import BENCHMARK_BBOX, BENCHMARK_CRS


@pytest.mark.benchmark
def test_open_overhead(benchmark, benchmark_parquet: str) -> None:
    """Phase 0: time the open() call without triggering any COG reads.

    Measures parquet queries, band discovery, time-step building, and grid
    computation.
    """
    benchmark(
        lazycogs.open,
        benchmark_parquet,
        bbox=BENCHMARK_BBOX,
        crs=BENCHMARK_CRS,
        resolution=60.0,
    )


@pytest.mark.benchmark
def test_full_compute(benchmark, benchmark_parquet: str) -> None:
    """Full pipeline: open + .compute() including local COG I/O."""

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
        )
        return da.compute()

    benchmark(run)


@pytest.mark.benchmark
@pytest.mark.parametrize("method", [FirstMethod, MedianMethod], ids=["first", "median"])
def test_mosaic_method(
    benchmark, benchmark_parquet: str, method: type[MosaicMethodBase]
) -> None:
    """Compare mosaic strategy cost end-to-end."""

    def run() -> object:
        da = lazycogs.open(
            benchmark_parquet,
            bbox=BENCHMARK_BBOX,
            crs=BENCHMARK_CRS,
            resolution=60.0,
            time_period="P1M",
            mosaic_method=method,
        )
        return da.compute()

    benchmark(run)
