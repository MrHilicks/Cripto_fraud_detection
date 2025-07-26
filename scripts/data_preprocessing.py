# scripts/data_preprocessing.py

import os
import json
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    AVAILABLE_TS_FEATURES = {
        'borrow_timestamp': {'day', 'dayofyear', 'week'},
        'first_tx_timestamp': {'day', 'dayofweek', 'dayofyear', 'hour', 'hour_cos', 'hour_sin', 'minute', 'second', 'week'},
        'last_tx_timestamp': {'week', 'dayofyear'},
        'risky_first_tx_timestamp': {'day', 'dayofweek', 'dayofyear', 'hour', 'hour_cos', 'hour_sin', 'minute', 'second', 'week'},
        'risky_last_tx_timestamp': {'dayofyear'},
    }

    def __init__(
        self,
        timestamp_columns=None,
        timestamp_features=None,
        numeric_columns=None,
        quantile_output_distribution='normal',
        quantile_n_quantiles=1000,
        random_state=42
    ):
        self.timestamp_columns = timestamp_columns or [
            'borrow_timestamp',
            'first_tx_timestamp',
            'last_tx_timestamp',
            'risky_first_tx_timestamp',
            'risky_last_tx_timestamp'
        ]
        self.timestamp_features = timestamp_features if timestamp_features else self.AVAILABLE_TS_FEATURES
        self.numeric_columns = numeric_columns or ['repay_amount_sum_eth']
        self.quantile_output_distribution = quantile_output_distribution
        self.quantile_n_quantiles = quantile_n_quantiles
        self.random_state = random_state
        self.quantile_transformers = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        if self.numeric_columns is None:
            self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.difference(self.timestamp_columns).tolist()
        else:
            self.numeric_columns_ = [col for col in self.numeric_columns if col not in self.timestamp_columns]

        for col in self.numeric_columns_:
            qt = QuantileTransformer(
                output_distribution=self.quantile_output_distribution,
                n_quantiles=min(self.quantile_n_quantiles, X.shape[0]),
                random_state=self.random_state
            )
            qt.fit(X[[col]])
            self.quantile_transformers[col] = qt

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = pd.DataFrame(index=X.index)

        for col in self.numeric_columns_:
            qt = self.quantile_transformers.get(col)
            if qt:
                result[f'{col}_qt'] = qt.transform(X[[col]]).flatten()

        for col in self.timestamp_columns:
            ts = pd.to_datetime(X[col], unit='s')
            prefix = col
            temp = {}

            if 'day' in self.timestamp_features[col]:
                temp[f'{prefix}_day'] = ts.dt.day
            if 'dayofweek' in self.timestamp_features[col]:
                temp[f'{prefix}_dayofweek'] = ts.dt.dayofweek
            if 'dayofyear' in self.timestamp_features[col]:
                temp[f'{prefix}_dayofyear'] = ts.dt.dayofyear
            if 'week' in self.timestamp_features[col]:
                temp[f'{prefix}_week'] = ts.dt.isocalendar().week
            if 'hour' in self.timestamp_features[col]:
                temp[f'{prefix}_hour'] = ts.dt.hour
            if 'minute' in self.timestamp_features[col]:
                temp[f'{prefix}_minute'] = ts.dt.minute
            if 'second' in self.timestamp_features[col]:
                temp[f'{prefix}_second'] = ts.dt.second
            if 'hour_sin' in self.timestamp_features[col]:
                temp[f'{prefix}_hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
            if 'hour_cos' in self.timestamp_features[col]:
                temp[f'{prefix}_hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)

            temp_df = pd.DataFrame(temp)
            result = pd.concat([result, temp_df], axis=1)

        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return cloudpickle.load(f)


class DataPreprocessor:
    COLUMNS = [
        'repay_amount_sum_eth',
        'risk_factor',
        'max_risk_factor',
        'avg_risk_factor',
        'total_available_borrows_avg_eth',
        'time_since_first_deposit',
        'borrow_block_number',
        'risk_factor_above_threshold_daily_count',
        'market_atr',
        'borrow_count',
        'wallet_age',
        'borrow_amount_avg_eth',
        'repay_count',
        'min_eth_ever',
        'deposit_amount_sum_eth',
        'total_available_borrows_eth',
        'incoming_tx_avg_eth',
        'avg_weighted_risk_factor',
        'total_collateral_avg_eth',
        'withdraw_amount_sum_eth',
        'market_natr',
        'market_adxr',
        'risky_first_tx_timestamp',
        'risky_tx_count',
        'outgoing_tx_count',
        'incoming_tx_count',
        'risky_sum_outgoing_amount_eth',
        'market_aroonosc',
        'risky_unique_contract_count',
        'market_macdsignal_macdfix',
        'max_eth_ever',
        'deposit_count',
        'total_balance_eth',
        'time_since_last_liquidated',
        'market_plus_dm',
        'repay_amount_avg_eth',
        'first_tx_timestamp',
        'total_collateral_eth',
        'total_gas_paid_eth',
        'risky_first_last_tx_timestamp_diff',
        'borrow_repay_diff_eth',
        'liquidation_amount_sum_eth',
        'outgoing_tx_sum_eth',
        'outgoing_tx_avg_eth',
        'liquidation_count',
        'incoming_tx_sum_eth',
        'market_macd_macdfix',
        'borrow_amount_sum_eth',
        'market_apo',
        'market_linearreg_slope',
        'withdraw_deposit_diff_if_positive_eth',
        'market_cmo',
        'unique_lending_protocol_count',
        'market_macdsignal_macdext',
        'unique_borrow_protocol_count',
        'market_adx',
        'market_cci',
        'market_fastk',
    ]

    def data_preprocessing(self, df: pd.DataFrame, feature_preprocessor: FeaturePreprocessor) -> pd.DataFrame:
        return pd.concat([df, feature_preprocessor.transform(df)], axis=1)[self.COLUMNS]
