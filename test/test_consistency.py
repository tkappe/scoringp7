import pytest
import pandas as pd

def test_train_target_col(data_train):
    """Test that the train dataframe has a 'target' column"""
    assert 'TARGET' in data_train.columns


def test_train_test_sizes(data_train, data_test):
    """Check that train and test dataframe have the same columns (but target)"""
    train_size = data_train.drop(columns='TARGET').shape[1]
    test_size = data_test.shape[1]
    assert train_size == test_size
    
