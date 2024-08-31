import pytest
import pandas as pd

def test_train_duplicates(data_train):
    """Test if the train duplicated dataframe is empty --> no duplicates"""
    duplicates = data_train[data_train.duplicated()]
    assert duplicates.empty


def test_test_duplicates(data_test):
    """Test if the test duplicated dataframe is empty --> no duplicates"""
    duplicates = data_test[data_test.duplicated()]
    assert duplicates.empty
    