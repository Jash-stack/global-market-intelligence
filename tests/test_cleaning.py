"""Tests for the data cleaning module."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_ohlcv():
    """Minimal OHLCV dataframe with missing values."""
    return pd.DataFrame({
        "open": [100.0, np.nan, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, np.nan, 108.0, 109.0],
        "low":  [95.0,  96.0,  97.0,  np.nan, 99.0],
        "close":[101.0, 102.0, 103.0, 104.0, np.nan],
        "volume":[1000, 1100, 1200, 1300, 1400],
    }, index=pd.date_range("2024-01-01", periods=5, freq="D"))


@pytest.fixture
def df_with_duplicates(raw_ohlcv):
    return pd.concat([raw_ohlcv, raw_ohlcv.iloc[:2]])


class TestMissingValueHandling:
    def test_forward_fill_removes_nans(self, raw_ohlcv):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(raw_ohlcv)
        assert cleaned.isnull().sum().sum() == 0

    def test_no_extra_rows_added(self, raw_ohlcv):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(raw_ohlcv)
        assert len(cleaned) == len(raw_ohlcv)

    def test_volume_unchanged(self, raw_ohlcv):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(raw_ohlcv)
        pd.testing.assert_series_equal(
            cleaned["volume"].reset_index(drop=True),
            raw_ohlcv["volume"].reset_index(drop=True),
        )


class TestDeduplication:
    def test_duplicates_removed(self, df_with_duplicates):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(df_with_duplicates)
        assert not cleaned.index.duplicated().any()

    def test_row_count_correct(self, raw_ohlcv, df_with_duplicates):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(df_with_duplicates)
        assert len(cleaned) == len(raw_ohlcv)


class TestDataTypes:
    def test_numeric_columns_are_float(self, raw_ohlcv):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(raw_ohlcv)
        for col in ["open", "high", "low", "close"]:
            assert cleaned[col].dtype == np.float64

    def test_index_is_datetime(self, raw_ohlcv):
        from src.cleaning import clean_ohlcv
        cleaned = clean_ohlcv(raw_ohlcv)
        assert isinstance(cleaned.index, pd.DatetimeIndex)
