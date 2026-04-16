"""stac-cog-xarray: lazy xarray DataArrays from STAC COG collections."""

from stac_cog_xarray._core import open, open_async
from stac_cog_xarray._explain import (  # noqa: F401 — registers da.stac_cog accessor
    ChunkRead,
    ExplainPlan,
    CogRead,
    StacCogAccessor,
)
from stac_cog_xarray._mosaic_methods import (
    CountMethod,
    FirstMethod,
    HighestMethod,
    LowestMethod,
    MeanMethod,
    MedianMethod,
    MosaicMethodBase,
    StdevMethod,
)

__all__ = [
    "open",
    "open_async",
    "ExplainPlan",
    "ChunkRead",
    "CogRead",
    "MosaicMethodBase",
    "FirstMethod",
    "HighestMethod",
    "LowestMethod",
    "MeanMethod",
    "MedianMethod",
    "StdevMethod",
    "CountMethod",
]
