import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self, filepath='data/raw/india_city_property_data.csv'):
        """Load raw data and perform initial cleaning"""
        if not os.path.exists(filepath):
            print(f"⚠️ Data file not found: {filepath}")
            print("📊 Generating data first...")
            from data_generator import IndianPropertyDataGenerator
            generator = IndianPropertyDataGenerator()
            generator.save_data()
        
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} records")
        
        # Basic cleaning
        df = df.dropna()
        
        # Remove outliers
        for col in ['sale_price', 'rent_price']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        print(f"✅ After cleaning: {len(df)} records")
        return df
    
    def preprocess(self, df):
        """Feature engineering and preprocessing"""
        # Create copy
        df = df.copy()
        
        # Feature engineering
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['price_per_sqft'] = df['sale_price'] / df['area']
        df['rent_per_sqft'] = df['rent_price'] / df['area']
        
        # Encoding categorical variables
        categorical_cols = ['city', 'property_type', 'furnishing', 'property_age']
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.encoders[col].fit_transform(df[col])
        
        # Select features
        features = ['city_encoded', 'property_type_encoded', 'area', 'bedrooms', 
                   'bathrooms', 'balcony', 'hall', 'furnishing_encoded', 
                   'property_age_encoded', 'total_rooms']
        
        X = df[features]
        y_sale = df['sale_price']
        y_rent = df['rent_price']
        
        return X, y_sale, y_rent
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        numerical_cols = ['area', 'total_rooms']
        X_scaled = X.copy()
        
        if fit:
            X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
            
        return X_scaled
    
    def save_preprocessor(self, path='data/models'):
        """Save preprocessor objects"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.encoders, f'{path}/encoders.pkl')
        print(f"✅ Preprocessor saved to {path}/")
    
    def load_preprocessor(self, path='data/models'):
        """Load preprocessor objects"""
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.encoders = joblib.load(f'{path}/encoders.pkl')
        print(f"✅ Preprocessor loaded from {path}/")