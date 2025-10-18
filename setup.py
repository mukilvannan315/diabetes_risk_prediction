

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print(" Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error installing packages: {e}")
        return False

def train_model():
    """Train the machine learning model"""
    print(" Training the machine learning model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print(" Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error training model: {e}")
        return False

def main():
    """Main setup function"""
    print(" Diabetes Risk Prediction System Setup")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists("diabetes_prediction_dataset.csv"):
        print(" Dataset file not found!")
        print("Please ensure 'diabetes_prediction_dataset.csv' is in the current directory.")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Train model
    if not train_model():
        return False
    
    print("\n Setup completed successfully!")
    print("\nTo run the application:")
    print("  streamlit run app.py")
    print("\nThen open your browser to: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main()

