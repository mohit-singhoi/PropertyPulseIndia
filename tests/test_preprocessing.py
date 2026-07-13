import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Add parent directory to path to import data_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None  # Store feature names
        
    def load_data(self, filepath='data/raw/india_city_property_data.csv'):
        """Load raw data and perform initial cleaning"""
        # Check if file exists, if not generate data
        if not os.path.exists(filepath):
            print(f"⚠️ Data file not found: {filepath}")
            print("📊 Generating data first...")
            try:
                from data_generator import IndianPropertyDataGenerator
                generator = IndianPropertyDataGenerator()
                generator.save_data()
                print("✅ Data generated successfully!")
            except ImportError as e:
                print(f"❌ Error: Could not import data_generator: {e}")
                print("Please run 'python data_generator.py' first")
                return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} records from {filepath}")
            
            # Basic cleaning
            df = df.dropna()
            
            # Remove outliers for numerical columns
            for col in ['sale_price', 'rent_price', 'area']:
                if col in df.columns:
                    Q1 = df[col].quantile(0.01)
                    Q3 = df[col].quantile(0.99)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            print(f"✅ After cleaning: {len(df)} records")
            return df
            
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            print("Please run 'python data_generator.py' to generate data")
            return None
    
    def preprocess(self, df):
        """Feature engineering and preprocessing"""
        if df is None or len(df) == 0:
            print("❌ No data to preprocess")
            return None, None, None
            
        # Create copy
        df = df.copy()
        
        # Feature engineering
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['price_per_sqft'] = df['sale_price'] / df['area']
        df['rent_per_sqft'] = df['rent_price'] / df['area']
        
        # Encoding categorical variables
        categorical_cols = ['city', 'property_type', 'furnishing', 'property_age']
        for col in categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
        
        # Select features - STORE THIS ORDER for later use
        self.feature_names = ['city_encoded', 'property_type_encoded', 'area', 'bedrooms', 
                             'bathrooms', 'balcony', 'hall', 'furnishing_encoded', 
                             'property_age_encoded', 'total_rooms']
        
        # Check if all features exist
        available_features = [f for f in self.feature_names if f in df.columns]
        if len(available_features) < len(self.feature_names):
            print(f"⚠️ Missing features: {set(self.feature_names) - set(available_features)}")
            self.feature_names = available_features
        
        X = df[self.feature_names]
        y_sale = df['sale_price'] if 'sale_price' in df.columns else None
        y_rent = df['rent_price'] if 'rent_price' in df.columns else None
        
        return X, y_sale, y_rent
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if X is None or len(X) == 0:
            return X
            
        numerical_cols = ['area', 'total_rooms']
        numerical_cols = [col for col in numerical_cols if col in X.columns]
        
        X_scaled = X.copy()
        
        if numerical_cols:
            if fit:
                X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            else:
                X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
                
        return X_scaled
    
    def save_preprocessor(self, path='data/models'):
        """Save preprocessor objects including feature names"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.encoders, f'{path}/encoders.pkl')
        joblib.dump(self.feature_names, f'{path}/feature_names.pkl')  # Save feature names
        print(f"✅ Preprocessor saved to {path}/")
        print(f"✅ Feature names saved: {self.feature_names}")
    
    def load_preprocessor(self, path='data/models'):
        """Load preprocessor objects"""
        try:
            self.scaler = joblib.load(f'{path}/scaler.pkl')
            self.encoders = joblib.load(f'{path}/encoders.pkl')
            self.feature_names = joblib.load(f'{path}/feature_names.pkl')
            print(f"✅ Preprocessor loaded from {path}/")
            print(f"✅ Feature names loaded: {self.feature_names}")
            return True
        except FileNotFoundError:
            print(f"❌ Preprocessor files not found in {path}/")
            return False

# Quick test
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    if df is not None:
        X, y_sale, y_rent = preprocessor.preprocess(df)
        if X is not None:
            X_scaled = preprocessor.scale_features(X)
            print(f"✅ Features shape: {X_scaled.shape}")
            print(f"✅ Feature names: {preprocessor.feature_names}")
            if y_sale is not None:
                print(f"✅ Sale price range: {y_sale.min():,.0f} - {y_sale.max():,.0f}")
            if y_rent is not None:
                print(f"✅ Rent price range: {y_rent.min():,.0f} - {y_rent.max():,.0f}")
            preprocessor.save_preprocessor()