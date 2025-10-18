

import pandas as pd
import numpy as np
import joblib
import os

def test_model_loading():
    """Test if model files can be loaded"""
    print("🧪 Testing model loading...")
    
    try:
        model = joblib.load('diabetes_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        selected_features = joblib.load('selected_features.pkl')
        print(" Model files loaded successfully!")
        return model, label_encoders, selected_features
    except FileNotFoundError as e:
        print(f" Model files not found: {e}")
        print("Please run 'python train_model.py' first")
        return None, None, None
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None, None

def test_prediction(model, label_encoders, selected_features):
    """Test prediction with sample data"""
    print("Testing prediction...")
    
    # Sample test data
    test_data = {
        'gender': ['Male'],
        'age': [45],
        'hypertension': [0],
        'heart_disease': [0],
        'smoking_history': ['never'],
        'bmi': [25.5],
        'HbA1c_level': [5.2],
        'blood_glucose_level': [95]
    }
    
    try:
        # Create DataFrame
        df = pd.DataFrame(test_data)
        
        # Encode categorical variables using the appropriate encoders
        df['gender'] = label_encoders['gender'].transform(df['gender'])
        df['smoking_history'] = label_encoders['smoking_history'].transform(df['smoking_history'])
        
        # Select features
        df_selected = df[selected_features]
        
        # Convert to numpy array to avoid feature name warnings
        X_test = df_selected.values
        
        # Make prediction
        probability = model.predict_proba(X_test)[0][1]
        prediction = model.predict(X_test)[0]
        
        print(f" Prediction successful!")
        print(f"   Probability: {probability:.3f}")
        print(f"   Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        
        return True
        
    except Exception as e:
        print(f" Prediction failed: {e}")
        return False

def main():
    """Main test function"""
    print(" Diabetes Prediction System - Test Suite")
    print("=" * 50)
    
    # Test model loading
    model, label_encoders, selected_features = test_model_loading()
    
    if model is None:
        print("\nTests failed - model not available")
        return False
    
    # Test prediction
    if not test_prediction(model, label_encoders, selected_features):
        print("\n Tests failed - prediction error")
        return False
    
  
    return True

if __name__ == "__main__":
    main()
