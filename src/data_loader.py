import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CashbackDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = None

    def _clean_data(self, df):
        """Clean data from infinite values, NaNs and extreme values"""
        logger.info("Cleaning data from infinite and NaN values...")

        df_clean = df.copy()

        inf_count_before = (
            np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        )
        nan_count_before = df_clean.isna().sum().sum()

        logger.info(
            f"Found {inf_count_before} inf val and {nan_count_before} NaN values"
        )

        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

            if df_clean[col].dtype in [np.float64, np.float32]:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

        inf_count_after = (
            np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
        )
        nan_count_after = df_clean.isna().sum().sum()

        logger.info(
            f"After cleaning: {inf_count_after} inf val and {nan_count_after} NaN"
        )

        return df_clean

    def _validate_data(self, df, dataset_name):
        """Validate data quality"""
        logger.info(f"Validating {dataset_name} data...")

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        has_inf = np.isinf(df[numerical_cols]).any().any()
        has_nan = df[numerical_cols].isna().any().any()

        if has_inf or has_nan:
            logger.warning(f"{dataset_name} still contains infinite or NaN values!")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median(numeric_only=True))

        for col in numerical_cols[:5]:
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()
            logger.debug(
                f"{col}: min={col_min:.4f}, max={col_max:.4f}, mean={col_mean:.4f}"
            )

        return df

    def load_and_preprocess_data(self):
        """Load and preprocess train and test data from parquet files"""
        logger.info("Loading parquet files...")
        train_df = pd.read_parquet(self.config["data"]["train_path"]).sample(
            n=1000000, random_state=self.config["data"]["random_seed"]
        )
        test_df = pd.read_parquet(self.config["data"]["test_path"]).sample(
            n=1000000, random_state=self.config["data"]["random_seed"]
        )

        logger.info(f"Raw train data shape: {train_df.shape}")
        logger.info(f"Raw test data shape: {test_df.shape}")

        train_df = self._clean_data(train_df)
        test_df = self._clean_data(test_df)

        train_df = self._validate_data(train_df, "train")
        test_df = self._validate_data(test_df, "test")

        exclude_cols = self.config["data"]["exclude_columns"] + [
            self.config["data"]["target_column"]
        ]
        self.feature_columns = [
            col for col in train_df.columns if col not in exclude_cols
        ]

        logger.info(f"Using {len(self.feature_columns)} features")
        logger.info(f"First 10 features: {self.feature_columns[:10]}")

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.config["data"]["target_column"]]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.config["data"]["target_column"]]

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        train_class_counts = y_train.value_counts()
        test_class_counts = y_test.value_counts()
        logger.info(f"Train class distribution: {dict(train_class_counts)}")
        logger.info(f"Test class distribution: {dict(test_class_counts)}")

        logger.info("Final data validation before scaling...")
        X_train = self._validate_data(X_train, "X_train")
        X_test = self._validate_data(X_test, "X_test")

        logger.info("Scaling features...")
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("Feature scaling completed successfully")
            joblib.dump(self.scaler, "scaler.pkl")
            print("✅ Scaler saved")
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            logger.info("Trying alternative scaling approach...")

            X_train_array = X_train.to_numpy()
            X_test_array = X_test.to_numpy()

            median = np.median(X_train_array, axis=0)
            iqr = np.percentile(X_train_array, 75, axis=0) - np.percentile(
                X_train_array, 25, axis=0
            )

            iqr[iqr == 0] = 1.0

            X_train_scaled = (X_train_array - median) / iqr
            X_test_scaled = (X_test_array - median) / iqr

            X_train_scaled = np.clip(X_train_scaled, -10, 10)
            X_test_scaled = np.clip(X_test_scaled, -10, 10)

            logger.info("Alternative scaling completed")

        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled,
            y_train,
            test_size=self.config["data"]["validation_size"],
            random_state=self.config["data"]["random_seed"],
            stratify=y_train,
        )

        logger.info(f"Final train shape: {X_train_final.shape}")
        logger.info(f"Validation shape: {X_val.shape}")
        logger.info(f"Test shape: {X_test_scaled.shape}")

        self._check_final_data_quality(X_train_final, "X_train_final")
        self._check_final_data_quality(X_val, "X_val")
        self._check_final_data_quality(X_test_scaled, "X_test_scaled")

        return (X_train_final, y_train_final, X_val, y_val, X_test_scaled, y_test)

    def _check_final_data_quality(self, data, name):
        """Check final data quality"""
        if isinstance(data, np.ndarray):
            has_inf = np.isinf(data).any()
            has_nan = np.isnan(data).any()
            data_min = np.min(data)
            data_max = np.max(data)
            data_mean = np.mean(data)

            logger.info(
                f"{name} - Inf: {has_inf}, NaN: {has_nan}, "
                f"Range: [{data_min:.4f}, {data_max:.4f}], Mean: {data_mean:.4f}"
            )
        else:
            logger.warning(f"Cannot check data quality for {name}, type: {type(data)}")

    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create PyTorch DataLoaders"""
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        train_dataset = CashbackDataset(X_train, y_train)
        val_dataset = CashbackDataset(X_val, y_val)
        test_dataset = CashbackDataset(X_test, y_test)

        batch_size = self.config["training"]["batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Created data loaders with batch_size={batch_size}")
        logger.info(
            f"Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
        )

        return train_loader, val_loader, test_loader
