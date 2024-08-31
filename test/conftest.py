import os
import pytest
import pandas as pd

@pytest.fixture(scope='module')
def data_train():
    """Get customers processed train data to feed into the tests"""
    data_train = pd.read_csv("./data/data_train_sampled.csv")
    return data_train


@pytest.fixture(scope='module')
def data_test():
    """Get customers processed test data to feed into the tests"""
    data_test = pd.read_csv("./data/data_test_sampled.csv")
    return data_test
