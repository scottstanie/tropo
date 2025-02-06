import numpy as np
import xarray as xr
import pytest
from RAiDER.models import HRES

TEST_DATA = 'data/test.nc'
PRESSURE = 'data/pressure.npy'
HGT = 'data/pressure.npy' 

# Load data
@pytest.fixture
def load_data():
    ds = xr.open_dataset(TEST_DATA)
    gold_pres = np.load(PRESSURE)
    gold_hgt = np.load(HGT)
    
    # Initialize HRES model and extract necessary values
    hres_model = HRES()
    hres_model._t = ds.t.isel(time=0).values
    hres_model._q = ds.q.isel(time=0).values

    return hres_model, ds, gold_pres, gold_hgt

def test_raider_geoh(load_data):
    hres_model, ds, gold_pres, gold_hgt = load_data

    # Perform the calculation
    _, pres, hgt = hres_model._calculategeoh(
        ds.z.isel(time=0, level=0).values,
        ds.lnsp.isel(time=0, level=0).values
    )

    # Check if results match the golden data
    np.testing.assert_array_almost_equal(pres, gold_pres)
    np.testing.assert_array_almost_equal(hgt, gold_hgt)
    