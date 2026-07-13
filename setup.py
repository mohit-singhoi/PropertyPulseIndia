import os
import subprocess
import sys

def create_project_structure():
    """Create the complete project folder structure"""
    folders = [
        'data/raw', 
        'data/processed', 
        'data/models',
        'src', 
        'notebooks', 
        'assets/images', 
        'assets/css',
        'assets/icons', 
        'tests'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    # Create __init__.py in src
    with open('src/__init__.py', 'w') as f:
        f.write('# House Price Prediction Package\n')
    
    print("✅ Project structure created successfully!")

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("✅ Dependencies installed successfully!")

def generate_data():
    """Generate sample data"""
    print("\n📊 Generating sample data...")
    from data_generator import IndianPropertyDataGenerator
    generator = IndianPropertyDataGenerator()
    generator.save_data()
    print("✅ Sample data generated!")

def train_models():
    """Train initial models"""
    print("\n🏠 Training models...")
    from src.model_training import ModelTrainer
    from src.data_preprocessing import DataPreprocessor
    from sklearn.model_selection import train_test_split
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X, y_sale, y_rent = preprocessor.preprocess(df)
    X_scaled = preprocessor.scale_features(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_sale, test_size=0.2, random_state=42
    )
    
    # Train model
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train, X_val, y_val)
    trainer.save_model(trainer.best_model, 'data/models/sale_price_model.pkl')
    
    # Train rent model
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
        X_scaled, y_rent, test_size=0.2, random_state=42
    )
    
    trainer_r = ModelTrainer()
    trainer_r.train_models(X_train_r, y_train_r, X_val_r, y_val_r)
    trainer_r.save_model(trainer_r.best_model, 'data/models/rent_price_model.pkl')
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("✅ Models trained successfully!")

if __name__ == "__main__":
    print("🚀 Setting up House Price Prediction System...")
    print("=" * 50)
    
    create_project_structure()
    install_requirements()
    generate_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nTo run the application:")
    print("1. Activate your virtual environment")
    print("2. Run: streamlit run app.py")
    print("\nTo train models:")
    print("Run: python src/model_training.py")