import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import sys
sys.path.append('src')

from data_loader import DataProcessor, CashbackDataset


class TestDataProcessor:
    """Тесты для DataProcessor"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            "data": {
                "train_path": "dummy_train.parquet",
                "test_path": "dummy_test.parquet", 
                "target_column": "active_essence",
                "exclude_columns": ["party_rk", "essence_id", "utilized"],
                "validation_size": 0.2,
                "random_seed": 42
            },
            "model": {
                "input_dim": 10
            },
            "training": {
                "batch_size": 32
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        """Создает тестовый DataFrame с разными типами данных"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'party_rk': range(n_samples),
            'essence_id': range(1000, 1000 + n_samples),
            'active_essence': np.random.randint(0, 2, n_samples),
            'utilized': np.random.randint(0, 2, n_samples),
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(10, 5, n_samples),
            'feature_3': np.random.exponential(2, n_samples)
        }
        
        data['feature_1'][0] = np.inf
        data['feature_2'][1] = -np.inf
        data['feature_3'][2] = np.nan
        
        return pd.DataFrame(data)
    
    def test_clean_data_removes_inf_and_nan(self, sample_config, sample_dataframe):
        """Тест очистки данных от infinite values и NaN"""
        processor = DataProcessor(sample_config)
        
        assert np.isinf(sample_dataframe.select_dtypes(include=[np.number])).sum().sum() > 0
        assert sample_dataframe.isna().sum().sum() > 0
        
        cleaned_df = processor._clean_data(sample_dataframe)
        
        assert not np.isinf(cleaned_df.select_dtypes(include=[np.number])).any().any()
        assert not cleaned_df.isna().any().any()
    
    def test_clean_data_preserves_structure(self, sample_config, sample_dataframe):
        """Тест, что очистка сохраняет структуру данных"""
        processor = DataProcessor(sample_config)
        cleaned_df = processor._clean_data(sample_dataframe)
        
        assert cleaned_df.shape == sample_dataframe.shape
        
        assert set(cleaned_df.columns) == set(sample_dataframe.columns)
    
    def test_validate_data_detects_problems(self, sample_config):
        """Тест валидации данных с проблемными значениями"""
        processor = DataProcessor(sample_config)
        
        problematic_data = pd.DataFrame({
            'feature_1': [1, 2, np.inf],
            'feature_2': [np.nan, 1, 2],
            'active_essence': [0, 1, 0]
        })
        
        validated_data = processor._validate_data(problematic_data, "test")
        
        assert not np.isinf(validated_data.select_dtypes(include=[np.number])).any().any()
        assert not validated_data.isna().any().any()
    
    @patch('pandas.read_parquet')
    def test_feature_selection(self, mock_read_parquet, sample_config, sample_dataframe):
        """Тест правильного выбора признаков"""
        mock_read_parquet.return_value = sample_dataframe
        processor = DataProcessor(sample_config)
        
        with patch.object(processor, '_clean_data', return_value=sample_dataframe):
            with patch.object(processor, '_validate_data', return_value=sample_dataframe):
                X_train, y_train, X_val, y_val, X_test, y_test = processor.load_and_preprocess_data()
        
        feature_cols = processor.feature_columns
        assert 'party_rk' not in feature_cols
        assert 'essence_id' not in feature_cols
        assert 'utilized' not in feature_cols
        assert 'active_essence' not in feature_cols
    
    def test_data_loader_creation(self, sample_config):
        """Тест создания DataLoader'ов"""
        processor = DataProcessor(sample_config)
        
        X_train = np.random.normal(0, 1, (100, 3))
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.normal(0, 1, (20, 3))
        y_val = np.random.randint(0, 2, 20)
        X_test = np.random.normal(0, 1, (30, 3))
        y_test = np.random.randint(0, 2, 30)
        
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        for batch in train_loader:
            features, targets = batch
            assert features.shape[0] == sample_config["training"]["batch_size"] or features.shape[0] == 100 % sample_config["training"]["batch_size"]
            assert targets.shape[0] == features.shape[0]
            break


class TestCashbackDataset:
    """Тесты для CashbackDataset"""
    
    def test_dataset_creation(self):
        """Тест создания датасета"""
        features = np.random.normal(0, 1, (100, 10))
        targets = np.random.randint(0, 2, 100)
        
        dataset = CashbackDataset(features, targets)
        
        assert len(dataset) == 100
        assert dataset.features.shape == (100, 10)
        assert dataset.targets.shape == (100,)
    
    def test_dataset_getitem(self):
        """Тест доступа к элементам датасета"""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = np.array([0, 1, 0])
        
        dataset = CashbackDataset(features, targets)
        
        for i in range(len(dataset)):
            feature, target = dataset[i]
            assert feature.shape == (2,)
            assert target.shape == ()
            assert feature.dtype == torch.float32
            assert target.dtype == torch.float32
    
    def test_dataset_edge_cases(self):
        """Тест граничных случаев датасета"""
        # Пустой датасет
        features = np.array([]).reshape(0, 5)
        targets = np.array([])
        
        dataset = CashbackDataset(features, targets)
        assert len(dataset) == 0
        
        features = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([1])
        
        dataset = CashbackDataset(features, targets)
        assert len(dataset) == 1
        feature, target = dataset[0]
        assert feature.shape == (3,)
        assert target.item() == 1.0


class TestDataLoaderEdgeCases:
    """Тесты граничных случаев и обработки ошибок"""
    
    def test_empty_dataframe(self, sample_config):
        """Тест обработки пустого DataFrame"""
        processor = DataProcessor(sample_config)
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            processor._clean_data(empty_df)
    
    def test_all_nan_data(self, sample_config):
        """Тест данных, состоящих полностью из NaN"""
        processor = DataProcessor(sample_config)
        
        nan_data = pd.DataFrame({
            'feature_1': [np.nan, np.nan, np.nan],
            'feature_2': [np.nan, np.nan, np.nan],
            'active_essence': [0, 1, 0]
        })
        
        cleaned_data = processor._clean_data(nan_data)
        assert not cleaned_data.isna().any().any()
    
    def test_extreme_values(self, sample_config):
        """Тест обработки экстремальных значений"""
        processor = DataProcessor(sample_config)
        
        extreme_data = pd.DataFrame({
            'feature_1': [1e100, -1e100, 0],
            'feature_2': [1e-100, -1e-100, 1],
            'active_essence': [0, 1, 0]
        })
        
        cleaned_data = processor._clean_data(extreme_data)
        
        assert cleaned_data['feature_1'].max() < 1e50
        assert cleaned_data['feature_1'].min() > -1e50