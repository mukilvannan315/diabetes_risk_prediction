
import os
import sys
import joblib
import pandas as pd
import numpy as np

def check_files():
    """Check if all required files exist"""
    print(" Checking required files...")
    
    required_files = [
        'diabetes_model.pkl',
        'label_encoders.pkl', 
        'selected_features.pkl',
        'diabetes_prediction_dataset.csv',
        'app.py',
        'train_model.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   {file}")
        else:
            print(f"   {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def check_model_loading():
    """Check if model and preprocessing objects can be loaded"""
    print("\n Checking model loading...")
    
    try:
        model = joblib.load('diabetes_model.pkl')
        print(f"   Model loaded - Type: {type(model).__name__}")
        
        encoders = joblib.load('label_encoders.pkl')
        print(f"   Encoders loaded - Keys: {list(encoders.keys())}")
        
        features = joblib.load('selected_features.pkl')
        print(f"   Features loaded - Count: {len(features)}")
        
        return True, model, encoders, features
        
    except Exception as e:
        print(f"   Model loading failed: {e}")
        return False, None, None, None

def check_prediction(model, encoders, features):
    """Check if prediction works"""
    print("\n Checking prediction system...")
    
    try:
        # Test data
        test_data = {
            'gender': ['Male'],
            'age': [45],
            'hypertension': [0],
            'heart_disease': [0],
            'smoking_history': ['never'],
            'bmi': [25.0],
            'HbA1c_level': [5.0],
            'blood_glucose_level': [100]
        }
        
        df = pd.DataFrame(test_data)
        
        # Encode categorical variables
        df['gender'] = encoders['gender'].transform(df['gender'])
        df['smoking_history'] = encoders['smoking_history'].transform(df['smoking_history'])
        
        # Select features
        df_selected = df[features]
        X_test = df_selected.values
        
        # Make prediction
        probability = model.predict_proba(X_test)[0][1]
        prediction = model.predict(X_test)[0]
        
        print(f"   Prediction successful")
        print(f"   Probability: {probability:.3f}")
        print(f"   Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        
        return True
        
    except Exception as e:
        print(f"   Prediction failed: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('joblib', 'joblib'),
        ('imbalanced-learn', 'imblearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"   {package_name}")
        except ImportError:
            print(f"   {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    return len(missing_packages) == 0, missing_packages

def main():
    """Main status check function"""
    print("🩺 Diabetes Risk Prediction System - Status Check")
    print("=" * 60)
    
    all_good = True
    
    # Check files
    files_ok, missing_files = check_files()
    if not files_ok:
        print(f"\n Missing files: {', '.join(missing_files)}")
        all_good = False
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"\n Missing packages: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        all_good = False
    
    # Check model loading
    if files_ok and deps_ok:
        model_ok, model, encoders, features = check_model_loading()
        if not model_ok:
            print("\n Model loading failed")
            all_good = False
        else:
            # Check prediction
            pred_ok = check_prediction(model, encoders, features)
            if not pred_ok:
                print("\n Prediction system failed")
                all_good = False
    
    # Final status
    print("\n" + "=" * 60)
    
    return all_good

if __name__ == "__main__":
    main()
