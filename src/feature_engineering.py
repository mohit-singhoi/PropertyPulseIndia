import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.features = {}
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df = df.copy()
        
        # Area to room ratio
        df['area_per_room'] = df['area'] / (df['bedrooms'] + df['bathrooms'])
        
        # Luxury score (based on amenities)
        df['luxury_score'] = (
            df['balcony'] * 0.2 + 
            df['hall'] * 0.3 +
            (df['furnishing'] == 'Furnished').astype(int) * 0.5
        )
        
        # Property age score (newer is better)
        age_mapping = {'New': 1.0, '1-5 years': 0.8, '5-10 years': 0.6, '10+ years': 0.4}
        df['age_score'] = df['property_age'].map(age_mapping)
        
        return df
    
    def create_location_features(self, df):
        """Create location-based features"""
        df = df.copy()
        
        # City tier (1, 2, 3)
        tier1_cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Hyderabad']
        tier2_cities = ['Pune', 'Noida', 'Gurugram', 'Ahmedabad']
        
        df['city_tier'] = df['city'].apply(
            lambda x: 1 if x in tier1_cities else (2 if x in tier2_cities else 3)
        )
        
        return df
    
    def create_all_features(self, df):
        """Create all engineered features"""
        df = self.create_interaction_features(df)
        df = self.create_location_features(df)
        return df