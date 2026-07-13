import joblib
import numpy as np
import pandas as pd
import os

def load_model(path):
    """Load saved model and preprocessors"""
    try:
        model = joblib.load(path)
        return model
    except:
        return None

def load_scaler_and_encoders():
    """Load scaler and encoders"""
    try:
        scaler = joblib.load('data/models/scaler.pkl')
        encoders = joblib.load('data/models/encoders.pkl')
        return scaler, encoders
    except:
        return None, None

def preprocess_input(input_data, encoders):
    """Preprocess user input to match training features"""
    df = pd.DataFrame([input_data])
    
    # Feature engineering - create same features as training
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Encode categorical variables
    categorical_cols = ['city', 'property_type', 'furnishing', 'property_age']
    for col in categorical_cols:
        if col in encoders:
            try:
                df[col + '_encoded'] = encoders[col].transform(df[col])
            except:
                # Handle unseen categories - use most frequent or default
                df[col + '_encoded'] = 0
    
    # Select features in the EXACT order as during training
    # This order MUST match what was used when fitting the scaler
    features = ['city_encoded', 'property_type_encoded', 'area', 'bedrooms', 
               'bathrooms', 'balcony', 'hall', 'furnishing_encoded', 
               'property_age_encoded', 'total_rooms']
    
    # Ensure all features exist
    available_features = []
    for feature in features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            # Create missing feature with default value
            df[feature] = 0
            available_features.append(feature)
    
    # Return dataframe with features in correct order
    return df[available_features]

def predict_price(input_data):
    """Predict sale price"""
    try:
        # Load models
        model = joblib.load('data/models/sale_price_model.pkl')
        scaler = joblib.load('data/models/scaler.pkl')
        encoders = joblib.load('data/models/encoders.pkl')
        
        # Preprocess input
        X = preprocess_input(input_data, encoders)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        confidence_interval = (prediction * 0.9, prediction * 1.1)
        
        return prediction, confidence_interval
        
    except Exception as e:
        print(f"Error in predict_price: {e}")
        # Fallback to simple calculation
        return predict_price_fallback(input_data)

def predict_rent(input_data):
    """Predict rental price"""
    try:
        # Load models
        model = joblib.load('data/models/rent_price_model.pkl')
        scaler = joblib.load('data/models/scaler.pkl')
        encoders = joblib.load('data/models/encoders.pkl')
        
        # Preprocess input
        X = preprocess_input(input_data, encoders)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        confidence_interval = (prediction * 0.9, prediction * 1.1)
        
        return prediction, confidence_interval
        
    except Exception as e:
        print(f"Error in predict_rent: {e}")
        # Fallback to simple calculation
        return predict_rent_fallback(input_data)

def predict_price_fallback(input_data):
    """Fallback prediction using city averages"""
    avg_prices = get_city_avg_prices(input_data['city'])
    
    base_price = avg_prices['sale']
    
    # Adjust based on features
    area_factor = input_data['area'] / 1200
    rooms_factor = 1 + (input_data['bedrooms'] - 3) * 0.1
    furnishing_factor = 1.15 if input_data['furnishing'] == 'Furnished' else (1.07 if input_data['furnishing'] == 'Semi-Furnished' else 1.0)
    
    predicted = base_price * area_factor * rooms_factor * furnishing_factor
    
    # Add some randomness
    predicted *= np.random.normal(1, 0.05)
    
    confidence_interval = (predicted * 0.85, predicted * 1.15)
    
    return predicted, confidence_interval

def predict_rent_fallback(input_data):
    """Fallback rent prediction using city averages"""
    avg_prices = get_city_avg_prices(input_data['city'])
    
    base_rent = avg_prices['rent']
    
    # Adjust based on features
    area_factor = input_data['area'] / 1200
    rooms_factor = 1 + (input_data['bedrooms'] - 3) * 0.08
    furnishing_factor = 1.2 if input_data['furnishing'] == 'Furnished' else (1.1 if input_data['furnishing'] == 'Semi-Furnished' else 1.0)
    
    predicted = base_rent * area_factor * rooms_factor * furnishing_factor
    
    # Add some randomness
    predicted *= np.random.normal(1, 0.05)
    
    confidence_interval = (predicted * 0.85, predicted * 1.15)
    
    return predicted, confidence_interval

def get_city_avg_prices(city):
    """Get average prices for a city"""
    city_data = {
        'Delhi': {'sale': 8500000, 'rent': 25000},
        'Mumbai': {'sale': 12000000, 'rent': 40000},
        'Bengaluru': {'sale': 7000000, 'rent': 22000},
        'Pune': {'sale': 5500000, 'rent': 18000},
        'Chennai': {'sale': 5000000, 'rent': 16000},
        'Hyderabad': {'sale': 6000000, 'rent': 19000},
        'Noida': {'sale': 6500000, 'rent': 20000},
        'Gurugram': {'sale': 7500000, 'rent': 24000},
        'Greater Noida': {'sale': 5500000, 'rent': 17000},
        'Jaipur': {'sale': 4500000, 'rent': 15000},
        'Ahmedabad': {'sale': 4800000, 'rent': 16000},
        'Lucknow': {'sale': 3800000, 'rent': 12000},
        'Kochi': {'sale': 5000000, 'rent': 17000},
        'Chandigarh': {'sale': 5500000, 'rent': 18000},
        'Patna': {'sale': 3500000, 'rent': 11000}
    }
    return city_data.get(city, {'sale': 5000000, 'rent': 15000})