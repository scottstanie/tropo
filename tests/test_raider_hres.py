import pickle
import pytest
import numpy as np
from RAiDER.models import HRES


@pytest.fixture
def load_model_and_dict():
    # Load the golden hres model dict from the file once
    with open('data/hres_model.pkl', 'rb') as file:
        golden_dict = pickle.load(file)
    
    # Initialize the HRES model
    hres_model = HRES()
    model_dict = hres_model.__dict__
    
    return golden_dict, model_dict


@pytest.mark.parametrize("key", [
    '_k1', '_k2', '_k3', '_humidityType', '_a', '_b', 
    '_model_level_type', '_R_v', '_R_d', '_g0', '_zmin', 
    '_zmax', '_zlevels'
])
def test_hres_class(load_model_and_dict, key):
    golden_dict, model_dict = load_model_and_dict
    
    # Check if key exists in both dictionaries
    assert key in golden_dict, f"Key '{key}' not found in golden_dict"
    assert key in model_dict, f"Key '{key}' not found in model_dict"
    
    # Get the values for the key
    golden_value = golden_dict[key]
    model_value = model_dict[key]
    
    # Handle comparison if values are arrays or lists
    if isinstance(golden_value, np.ndarray) and isinstance(model_value, np.ndarray):
        assert np.array_equal(golden_value, model_value), \
            f"Values for key '{key}' do not match: golden_dict = {golden_value}, model_dict = {model_value}"
    else:
        assert golden_value == model_value, \
            f"Values for key '{key}' do not match: golden_dict = {golden_value}, model_dict = {model_value}"


