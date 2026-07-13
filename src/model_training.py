from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
        }
        self.best_model = None
        self.best_score = -np.inf
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                r2 = r2_score(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }
                
                print(f"  ✅ R² Score: {r2:.4f}")
                print(f"  ✅ MAE: ₹{mae:,.2f}")
                print(f"  ✅ RMSE: ₹{rmse:,.2f}")
                print(f"  ✅ MAPE: {mape:.2f}%")
                
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest"""
        print("\n🔧 Performing hyperparameter tuning for Random Forest...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
    
    def save_model(self, model, path='data/models/sale_price_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='data/models/sale_price_model.pkl'):
        """Load trained model"""
        try:
            model = joblib.load(path)
            print(f"✅ Model loaded from {path}")
            return model
        except FileNotFoundError:
            print(f"❌ Model not found at {path}")
            return None

# Training script - this runs when you execute the file directly
if __name__ == "__main__":
    print("🏠 Training House Price Prediction Models...")
    print("=" * 60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    
    if df is None:
        print("❌ Failed to load data. Please run 'python data_generator.py' first.")
        sys.exit(1)
    
    # Preprocess data
    X, y_sale, y_rent = preprocessor.preprocess(df)
    
    if X is None or y_sale is None:
        print("❌ Failed to preprocess data.")
        sys.exit(1)
    
    # Scale features
    X_scaled = preprocessor.scale_features(X)
    
    print(f"\n📊 Data shape: {X_scaled.shape}")
    print(f"📊 Sale price range: ₹{y_sale.min():,.0f} - ₹{y_sale.max():,.0f}")
    print(f"📊 Rent price range: ₹{y_rent.min():,.0f} - ₹{y_rent.max():,.0f}")
    
    # Split data for sale price
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_sale, test_size=0.2, random_state=42
    )
    
    print("\n" + "=" * 60)
    print("📊 Training SALE PRICE Model...")
    print("=" * 60)
    
    # Train sale price model
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Print best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_result = results[best_model_name]
    print(f"\n🏆 Best Model for Sale Price: {best_model_name}")
    print(f"   R² Score: {best_result['r2']:.4f}")
    print(f"   MAE: ₹{best_result['mae']:,.2f}")
    print(f"   RMSE: ₹{best_result['rmse']:,.2f}")
    print(f"   MAPE: {best_result['mape']:.2f}%")
    
    # Save best sale price model
    trainer.save_model(trainer.best_model, 'data/models/sale_price_model.pkl')
    
    # Train rent price model
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
        X_scaled, y_rent, test_size=0.2, random_state=42
    )
    
    print("\n" + "=" * 60)
    print("📊 Training RENT PRICE Model...")
    print("=" * 60)
    
    trainer_r = ModelTrainer()
    results_r = trainer_r.train_models(X_train_r, y_train_r, X_val_r, y_val_r)
    
    # Print best model
    best_model_name_r = max(results_r, key=lambda x: results_r[x]['r2'])
    best_result_r = results_r[best_model_name_r]
    print(f"\n🏆 Best Model for Rent Price: {best_model_name_r}")
    print(f"   R² Score: {best_result_r['r2']:.4f}")
    print(f"   MAE: ₹{best_result_r['mae']:,.2f}")
    print(f"   RMSE: ₹{best_result_r['rmse']:,.2f}")
    print(f"   MAPE: {best_result_r['mape']:.2f}%")
    
    # Save best rent model
    trainer_r.save_model(trainer_r.best_model, 'data/models/rent_price_model.pkl')
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print("\n📁 Files saved:")
    print("  - data/models/sale_price_model.pkl")
    print("  - data/models/rent_price_model.pkl")
    print("  - data/models/scaler.pkl")
    print("  - data/models/encoders.pkl")
    print("\n🚀 To run the app:")
    print("  streamlit run app.py")