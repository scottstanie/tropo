#!/usr/bin/env python
import logging
from pathlib import Path
import xarray as xr

logger = logging.getLogger(__name__)

def compare_attrs(ds1: xr.Dataset, ds2: xr.Dataset):
    """Compares dictionary values while excluding 'history' key."""
    filtered_dict1 = {k: v for k, v in ds1.attrs.items() if k != "history"}
    filtered_dict2 = {k: v for k, v in ds2.attrs.items() if k != "history"}
    
    return filtered_dict1 == filtered_dict2

def compare_coord_attrs(ds1: xr.Dataset, ds2: xr.Dataset):
    """Compares coord attribues"""
    dict1 = {coord: ds1[coord].attrs for coord in ds1.coords}
    dict2 = {coord: ds2[coord].attrs for coord in ds2.coords}
    
    return dict1 == dict2

def compare_two_datasets(xr_file1 : str | Path, xr_file2 : str | Path) -> None:

    # Load two xarray Datasets
    ds1 = xr.open_dataset(xr_file1)
    ds2 = xr.open_dataset(xr_file2)

    # Compare dataset dimensions
    logger.info(f"Test Dataset dimensions")
    assert ds1.sizes == ds2.sizes, "Dataset dimensions do not match"

    #Compare variable names
    logger.info(f"Test Dataset Variable names")
    assert set(ds1.data_vars) == set(ds2.data_vars), "Variable names do not match"

    # Check global attrs
    logger.info(f"Test Dataset Global attributes")
    assert compare_attrs(ds1, ds2), "Global Attribute keys should match (ignoring 'history')"

    # Check coord attrs
    logger.info(f"Test Dataset Coordinate attributes")
    assert compare_coord_attrs(ds1, ds2), "Coords Attribute keys should match"

    # Raises an AssertionError if two objects are not equal. 
    #  This will match data values, dimensions and coordinates, 
    # but not names or attributes
    logger.info(f"Test Dataset Variable Data values")
    xr.testing.assert_equal(ds1, ds2)

    print(f"✅ Datasets {Path(xr_file1).name} and {Path(xr_file2).name} match!")