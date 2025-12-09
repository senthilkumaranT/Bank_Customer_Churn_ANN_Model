"""
Tests for the Streamlit app functionality.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAppImports:
    """Test that all required modules can be imported."""
    
    def test_import_app(self):
        """Test that app.py can be imported."""
        import app
        assert app is not None
    
    def test_import_required_modules(self):
        """Test that all required modules are available."""
        import streamlit as st
        import numpy as np
        import tensorflow as tf
        import pandas as pd
        import pickle
        import os
        assert True


class TestModelFiles:
    """Test that model files exist and can be loaded."""
    
    def test_model_files_exist(self):
        """Test that all required model files exist."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        
        assert os.path.exists(os.path.join(models_dir, 'model.h5')), "model.h5 not found"
        assert os.path.exists(os.path.join(models_dir, 'scaler.pkl')), "scaler.pkl not found"
        assert os.path.exists(os.path.join(models_dir, 'label_encoder_gender.pkl')), "label_encoder_gender.pkl not found"
        assert os.path.exists(os.path.join(models_dir, 'onehot_encoder_geo.pkl')), "onehot_encoder_geo.pkl not found"
    
    def test_load_model_files(self):
        """Test that model files can be loaded."""
        import tensorflow as tf
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        
        # Test loading model
        model_path = os.path.join(models_dir, 'model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            assert model is not None
        
        # Test loading scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            assert scaler is not None
        
        # Test loading encoders
        label_encoder_path = os.path.join(models_dir, 'label_encoder_gender.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            assert label_encoder is not None
        
        onehot_encoder_path = os.path.join(models_dir, 'onehot_encoder_geo.pkl')
        if os.path.exists(onehot_encoder_path):
            with open(onehot_encoder_path, 'rb') as f:
                onehot_encoder = pickle.load(f)
            assert onehot_encoder is not None


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_data_format(self):
        """Test that input data format is correct."""
        # Sample input data
        input_data = {
            'CreditScore': 650,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 35,
            'Tenure': 5,
            'Balance': 125000.50,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000.00
        }
        
        # Verify all required fields are present
        required_fields = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary'
        ]
        
        for field in required_fields:
            assert field in input_data, f"Missing required field: {field}"
        
        # Verify data types
        assert isinstance(input_data['CreditScore'], (int, float))
        assert isinstance(input_data['Age'], (int, float))
        assert isinstance(input_data['Tenure'], (int, float))
        assert isinstance(input_data['Balance'], (int, float))
        assert input_data['Geography'] in ['France', 'Germany', 'Spain']
        assert input_data['Gender'] in ['Male', 'Female']
        assert input_data['HasCrCard'] in [0, 1]
        assert input_data['IsActiveMember'] in [0, 1]


class TestRequirements:
    """Test that requirements.txt is valid."""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        requirements_path = os.path.join(base_dir, 'requirements.txt')
        assert os.path.exists(requirements_path), "requirements.txt not found"
    
    def test_requirements_installed(self):
        """Test that all requirements can be imported."""
        import streamlit
        import tensorflow
        import pandas
        import numpy
        import sklearn
        assert True

