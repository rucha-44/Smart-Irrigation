import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class IrrigationDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
        # StandardScaler for inputs (Temp, Wind, etc.)
        self.feature_scaler = StandardScaler()
        
        # MinMaxScaler for Target (Water Amount) 
        # This fixes the issue where predictions were stuck at "0" or "3.5"
        self.target_scaler = MinMaxScaler() 
        
        self.imputer = SimpleImputer(strategy='median') 
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        self.feature_columns = None

    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
        
        self.df = pd.read_csv(self.filepath, na_values=['', ' ', 'nan', 'NaN', 'null'])
        print(f"   > Raw Data Loaded: {self.df.shape}")
        return self.df

    def clean_data(self):
        self.df['WATER_NEEDED'] = pd.to_numeric(self.df['WATER_NEEDED'], errors='coerce')
        self.df['WATER_REQUIREMENT'] = pd.to_numeric(self.df['WATER_REQUIREMENT'], errors='coerce')

        # Strict Dropping
        self.df.dropna(subset=['WATER_NEEDED', 'WATER_REQUIREMENT'], inplace=True)

        # Consistency
        self.df.loc[self.df['WATER_NEEDED'] == 0, 'WATER_REQUIREMENT'] = 0.0
        self.df.loc[self.df['WATER_REQUIREMENT'] > 0.1, 'WATER_NEEDED'] = 1

        # Handle Features
        numerical_cols = ['TEMPERATURE', 'HUMIDITY', 'RAINFALL', 'WIND_SPEED', 'SOIL_MOISTURE']
        categorical_cols = ['CROP_TYPE', 'SOIL_TYPE', 'REGION', 'WEATHER_CONDITION']

        for col in numerical_cols:
             self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df[numerical_cols] = self.imputer.fit_transform(self.df[numerical_cols])
        self.df[categorical_cols] = self.cat_imputer.fit_transform(self.df[categorical_cols])

        # Clip Outliers
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.05); Q3 = self.df[col].quantile(0.95)
            self.df[col] = np.clip(self.df[col], Q1, Q3)

    def preprocess(self):
        # One-Hot Encoding
        categorical_cols = ['CROP_TYPE', 'SOIL_TYPE', 'REGION', 'WEATHER_CONDITION']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=False)
        self.feature_columns = [c for c in self.df.columns if c not in ['WATER_NEEDED', 'WATER_REQUIREMENT']]
        
        # Scale Features
        numerical_cols = ['TEMPERATURE', 'HUMIDITY', 'RAINFALL', 'WIND_SPEED', 'SOIL_MOISTURE']
        self.df[numerical_cols] = self.feature_scaler.fit_transform(self.df[numerical_cols])
        return self.df

    def split_data(self):
        X = self.df[self.feature_columns]
        y_cls = self.df['WATER_NEEDED'].values
        
        # Scale Target
        y_reg_raw = self.df['WATER_REQUIREMENT'].values.reshape(-1, 1)
        y_reg_scaled = self.target_scaler.fit_transform(y_reg_raw)
        
        X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = train_test_split(
            X, y_cls, y_reg_scaled, test_size=0.15, random_state=42, stratify=y_cls
        )
        
        return (
            X_train.values.astype('float32'), X_test.values.astype('float32'), 
            y_c_train.astype('float32'), y_c_test.astype('float32'), 
            y_r_train.astype('float32'), y_r_test.astype('float32')
        )

    def save_processors(self, path='models/saved/'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.feature_scaler, f'{path}feature_scaler.pkl')
        joblib.dump(self.target_scaler, f'{path}target_scaler.pkl')
        joblib.dump(self.feature_columns, f'{path}feature_columns.pkl')
        print("   > Processors Saved.")