"""Tests for the feature engineering module."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_series():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    close = pd.Series(100 + np.random.randn(60).cumsum(), index=dates, name="close")
    return close


@pytest.fixture
def ohlcv_df(price_series):
    df = pd.DataFrame({"close": price_series})
    df["open"]  = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"]  = df["close"] * 1.02
    df["low"]   = df["close"] * 0.98
    df["volume"] = np.random.randint(1000, 5000, len(df))
    return df


class TestLagFeatures:
    def test_lag_columns_created(self, ohlcv_df):
        from src.feature_engineering import add_lag_features
        result = add_lag_features(ohlcv_df, lags=[1, 3, 5])
        for lag in [1, 3, 5]:
            assert f"close_lag_{lag}" in result.columns

    def test_lag_values_correct(self, ohlcv_df):
        from src.feature_engineering import add_lag_features
        result = add_lag_features(ohlcv_df, lags=[1])
        valid = result.dropna()
        expected = ohlcv_df["close"].shift(1).dropna()
        np.testing.assert_allclose(
            valid["close_lag_1"].values, expected.values, rtol=1e-6
        )


class TestRollingFeatures:
    def test_rolling_mean_created(self, ohlcv_df):
        from src.feature_engineering import add_rolling_features
        result = add_rolling_features(ohlcv_df, windows=[5, 20])
        assert "close_ma_5" in result.columns
        assert "close_ma_20" in result.columns

    def test_rolling_std_created(self, ohlcv_df):
        from src.feature_engineering import add_rolling_features
        result = add_rolling_features(ohlcv_df, windows=[5])
        assert "close_std_5" in result.columns


class TestTechnicalIndicators:
    def test_rsi_in_range(self, ohlcv_df):
        from src.feature_engineering import add_rsi
        result = add_rsi(ohlcv_df, period=14)
        valid_rsi = result["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_macd_columns_exist(self, ohlcv_df):
        from src.feature_engineering import add_macd
        result = add_macd(ohlcv_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
