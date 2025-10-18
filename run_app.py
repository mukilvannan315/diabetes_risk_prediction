

import subprocess
import sys
import os

def check_model_files():
    
    required_files = ['diabetes_model.pkl', 'label_encoders.pkl', 'selected_features.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(" Model files not found!")
        print(f"Missing files: {', '.join(missing_files)}")
        print("\nPlease run the setup first:")
        print("  python setup.py")
        return False
    
    return True

def run_streamlit_app():
    #Run the Streamlit application
    print(" Starting Diabetes Risk Prediction System...")
    print("Opening in your default browser...")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f" Error running application: {e}")

def main():
    """Main run function"""
    print(" Diabetes Risk Prediction System")
    print("=" * 40)
    
    # Check if model files exist
    if not check_model_files():
        return
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
