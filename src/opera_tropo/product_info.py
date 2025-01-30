import numpy as np
from datetime import datetime, timezone
from numpy.typing import DTypeLike
from dataclasses import dataclass, field

import RAiDER

GLOBAL_ATTRS = {
    "Conventions": "CF-1.7",
    "title": "OPERA Zenith Troposphere Delay",
    "institution": "Jet Propulsion Laboratory (JPL)",
    "source": 'ECMWF, HRES model',
    "source_url": "https://www.ecmwf.int/en/forecasts/datasets/set-i",
    "history": str(datetime.now(timezone.utc)),
    "references":   "https://raider.readthedocs.io/en/latest/", 
    "description": ("OPERA One-way Tropospheric Zenith Delay, interpolate"
                   " with DEM and multiple with -4pi/radar wavelength (2 way)"
                   " to get the SAR correction"),
    "software": "RAiDER",
    "software_version": f"{RAiDER.__version__}",
    }


### DATASET COORDS ###
@dataclass
class ProductCoords:
    """Container for tropo xarray coord info."""
    axis : str
    units : str
    standard_name : str
    long_name: str
    description: str
    encoding : dict[str, str] = field(default_factory=dict) 
    
    @property
    def get_attr(self) -> dict:
        """Return all tropo coord  attrs excluding encoding"""
        out_dict = self.__dict__.copy()
        del out_dict['encoding']
        return out_dict
    

@dataclass
class TropoCoordAttrs:
    latitude: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="y",
            units="degrees",
            standard_name="latitude",
            long_name="latitude",
            description=("Angular distance of a point north or south"
                         " of the equator."),
            encoding={}
        )
    )
    longitude: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="x",
            units="degrees",
            standard_name="longitude",
            long_name="longitude",
            description=("Angular distance of a point east or west"
                         " of the Prime Meridian."),
            encoding={}
        )
    )
    height: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="z",
            units="m",
            standard_name="height",
            long_name="ellipsoidal_height",
            description="Height above ellipsoid WGS84",
            encoding={}
        )
    )
    time: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="t",
            units=None, # units specified in encoding
            standard_name="time",
            long_name="UTC time",
            description="Model base time",
            encoding={
                "units": "hours since 1900-01-01",
                "calendar": "gregorian"
            }
        )
    )
    @property
    def names(self) -> list[str]:
        """Return all tropo coord  as a list."""
        return [v for v in self.__dict__.keys()]
        

### DATASET VARIABLES ###
@dataclass
class ProductInfo:
    """Information about a troposphere product dataset."""

    name: str
    long_name: str
    description: str
    fillvalue: DTypeLike
    dtype: DTypeLike
    attrs: dict[str, str] = field(default_factory=dict)
    keep_bits: int | None = None

    def to_dict(self):
        desc_dict = dict(
            standard_name=self.name,
            long_name=self.long_name,
            description=self.description
        )
        return self.attrs | desc_dict


@dataclass
class TropoProducts:
    """Container for tropo product dataset info."""

    wet_delay: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="wet_delay",
            long_name="one way zenith wet delay",
            description=(
                "Total Zenith Wet Delay Correction at the top of the atmosphere"
            ),
            fillvalue=np.nan,
            # Note sure should I keep grid_mapping here
            attrs={"units": "meters",
                   "grid_mapping": "spatial_ref"},
            # 12 bits, for random values in meters from -1 to 1, has a max
            # quantization error of about 0.06 millimeters
            keep_bits=12,
            dtype=np.float32,
        )
    )

    hydrostatic_delay: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="hydrostatic_delay",
            long_name="one way zenith hydrostatic delay",
            description=(
                "Total Zenith Wet Delay Correction at the top of the atmosphere"
            ),
            fillvalue=np.nan,
            # Note sure should I keep grid_mapping here
            attrs={"units": "meters",
                   "grid_mapping": "spatial_ref"},
            # 12 bits, for random values in meters from -1 to 1, has a max
            # quantization error of about 0.06 millimeters
            keep_bits=12,
            dtype=np.float32,
        )
    )

    coords: TropoCoordAttrs = field(default_factory=TropoCoordAttrs,
                                    init=False)

    def __post_init__(self):
        self.coords = TropoCoordAttrs()

    def __iter__(self):
        """Return all tropo dataset info as an iterable."""
        return iter(self.__dict__.values())

    @property
    def names(self) -> list[str]:
        """Return all tropo dataset names as a list."""
        return [v.name for v in self.__dict__.values()]

# Create a single instance to be used throughout the application
TROPO_PRODUCTS = TropoProducts()