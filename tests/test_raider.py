import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from RAiDER.models import HRES

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA = TEST_DIR / "data/test_data.nc"
PRESSURE = TEST_DIR / "data/pressure.npy"
HGT = TEST_DIR / "data/hgt.npy"


# Load GEOH
@pytest.fixture
def load_geoh_data():
    ds = xr.open_dataset(TEST_DATA)
    gold_pres = np.load(PRESSURE)
    gold_hgt = np.load(HGT)

    # Initialize HRES model and extract necessary values
    hres_model = HRES()
    hres_model._t = ds.t.isel(time=0).values
    hres_model._q = ds.q.isel(time=0).values

    return hres_model, ds, gold_pres, gold_hgt


def test_raider_geoh(load_geoh_data):
    hres_model, ds, gold_pres, gold_hgt = load_geoh_data

    # Perform the calculation
    _, pres, hgt = hres_model._calculategeoh(
        ds.z.isel(time=0, level=0).values, ds.lnsp.isel(time=0, level=0).values
    )

    # Check if results match the golden data
    np.testing.assert_array_almost_equal(pres, gold_pres)
    np.testing.assert_array_almost_equal(hgt, gold_hgt)


@pytest.fixture
def load_hres_model():
    # Load the golden hres model dict from the file once
    with open(TEST_DIR / "data/hres_model.pkl", "rb") as file:
        golden_dict = pickle.load(file)

    # Initialize the HRES model
    hres_model = HRES()
    model_dict = hres_model.__dict__

    return golden_dict, model_dict


@pytest.mark.parametrize(
    "key",
    [
        "_k1",
        "_k2",
        "_k3",
        "_humidityType",
        "_a",
        "_b",
        "_model_level_type",
        "_R_v",
        "_R_d",
        "_g0",
        "_zmin",
        "_zmax",
        "_zlevels",
    ],
)
def test_hres_class(load_hres_model, key):
    golden_dict, model_dict = load_hres_model

    # Check if key exists in both dictionaries
    assert key in golden_dict, f"Key '{key}' not found in golden_dict"
    assert key in model_dict, f"Key '{key}' not found in model_dict"

    # Get the values for the key
    golden_value = golden_dict[key]
    model_value = model_dict[key]

    # Handle comparison if values are arrays or lists
    if isinstance(golden_value, np.ndarray) and isinstance(model_value, np.ndarray):
        assert np.array_equal(
            golden_value, model_value
        ), f"Values for key '{key}' do not match: golden_dict = {golden_value}, model_dict = {model_value}"
    else:
        assert (
            golden_value == model_value
        ), f"Values for key '{key}' do not match: golden_dict = {golden_value}, model_dict = {model_value}"
