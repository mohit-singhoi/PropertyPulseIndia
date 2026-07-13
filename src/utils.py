import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

def load_model(model_path):
    """
    Load a saved model from file
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded model object
    """
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            print(f"⚠️ Model not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def get_city_avg_prices(city):
    """
    Get average sale and rent prices for a city
    
    Args:
        city: City name
    
    Returns:
        Dictionary with 'sale' and 'rent' average prices
    """
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

def save_dataframe(df, filepath):
    """Save dataframe with proper format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.pkl'):
        df.to_pickle(filepath)
    elif filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.json'):
        df.to_json(filepath, orient='records')
    else:
        df.to_pickle(filepath)
    
    print(f"✅ Data saved to {filepath}")

def load_dataframe(filepath):
    """Load dataframe from file"""
    if filepath.endswith('.pkl'):
        return pd.read_pickle(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        return pd.read_pickle(filepath)

def format_currency(amount):
    """Format amount in Indian currency format"""
    return f"₹{amount:,.2f}"

def get_model_metrics(model, X_test, y_test):
    """Get model performance metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_test)
    
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

def get_city_insights(city):
    """Get insights for a specific city"""
    city_insights = {
        'Delhi': {
            'growth': 'High',
            'demand': 'Very High',
            'avg_price': 8500000,
            'trend': 'Upward',
            'description': 'Capital city with diverse real estate options'
        },
        'Mumbai': {
            'growth': 'Medium',
            'demand': 'Very High',
            'avg_price': 12000000,
            'trend': 'Stable',
            'description': 'Financial capital with premium property prices'
        },
        'Bengaluru': {
            'growth': 'Very High',
            'demand': 'High',
            'avg_price': 7000000,
            'trend': 'Upward',
            'description': 'IT hub with growing real estate market'
        },
        'Pune': {
            'growth': 'High',
            'demand': 'High',
            'avg_price': 5500000,
            'trend': 'Upward',
            'description': 'Educational and IT hub with affordable luxury'
        },
        'Hyderabad': {
            'growth': 'High',
            'demand': 'High',
            'avg_price': 6000000,
            'trend': 'Upward',
            'description': 'Growing IT and pharma hub'
        },
        'Chennai': {
            'growth': 'Medium',
            'demand': 'Medium',
            'avg_price': 5000000,
            'trend': 'Stable',
            'description': 'Automobile and IT hub with stable market'
        },
        'Noida': {
            'growth': 'High',
            'demand': 'High',
            'avg_price': 6500000,
            'trend': 'Upward',
            'description': 'Planned city with modern infrastructure'
        },
        'Gurugram': {
            'growth': 'High',
            'demand': 'High',
            'avg_price': 7500000,
            'trend': 'Upward',
            'description': 'Corporate hub with luxury properties'
        }
    }
    return city_insights.get(city, {
        'growth': 'Medium',
        'demand': 'Medium',
        'avg_price': 5000000,
        'trend': 'Stable',
        'description': 'Growing city with developing real estate market'
    })

def create_feature_summary(input_data, predicted_price):
    """Create summary of features and their impact"""
    summary = {
        'property_details': input_data,
        'predicted_price': predicted_price,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return summary

def save_prediction_history(prediction_data, filepath='data/prediction_history.json'):
    """Save prediction to history"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'r') as f:
            history = json.load(f)
    except:
        history = []
    
    history.append(prediction_data)
    
    # Keep only last 100 predictions
    if len(history) > 100:
        history = history[-100:]
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    return True

def load_prediction_history(filepath='data/prediction_history.json'):
    """Load prediction history"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return []

def calculate_price_trend(city, days=30):
    """
    Calculate price trend for a city
    
    Args:
        city: City name
        days: Number of days for trend calculation
    
    Returns:
        Dictionary with trend data
    """
    # Mock data - in production, this would come from a real data source
    base_prices = {
        'Delhi': 8500000,
        'Mumbai': 12000000,
        'Bengaluru': 7000000,
        'Pune': 5500000,
        'Hyderabad': 6000000,
        'Chennai': 5000000,
        'Noida': 6500000,
        'Gurugram': 7500000
    }
    
    base_price = base_prices.get(city, 5000000)
    
    # Generate some random trend
    trend = np.random.uniform(-5, 10)
    volatility = np.random.uniform(1, 3)
    
    return {
        'city': city,
        'current_price': base_price * (1 + trend / 100),
        'trend_percentage': trend,
        'volatility': volatility,
        'days': days
    }

def get_market_sentiment(city):
    """
    Get market sentiment for a city
    
    Args:
        city: City name
    
    Returns:
        Sentiment score and label
    """
    sentiments = {
        'Delhi': {'score': 75, 'label': 'Positive'},
        'Mumbai': {'score': 65, 'label': 'Neutral'},
        'Bengaluru': {'score': 85, 'label': 'Very Positive'},
        'Pune': {'score': 80, 'label': 'Positive'},
        'Hyderabad': {'score': 82, 'label': 'Positive'},
        'Chennai': {'score': 60, 'label': 'Neutral'},
        'Noida': {'score': 78, 'label': 'Positive'},
        'Gurugram': {'score': 80, 'label': 'Positive'}
    }
    
    return sentiments.get(city, {'score': 70, 'label': 'Neutral'})

# Ensure all required functions are exported
__all__ = [
    'load_model',
    'get_city_avg_prices',
    'save_dataframe',
    'load_dataframe',
    'format_currency',
    'get_model_metrics',
    'get_city_insights',
    'create_feature_summary',
    'save_prediction_history',
    'load_prediction_history',
    'calculate_price_trend',
    'get_market_sentiment'
]