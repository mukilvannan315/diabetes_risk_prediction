import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

def train_and_save_model():
    """Train the diabetes prediction model and save it"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    # Data preprocessing
    print("Preprocessing data...")
    df_encoded = df.copy()
    label_cols = ['gender', 'smoking_history']
    
    # Create separate label encoders for each column
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Feature selection
    X = df_encoded.drop('diabetes', axis=1)
    Y = df_encoded['diabetes']
    
    # Get top 5 features using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, Y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = importances.head(5).index.tolist()
    
    print(f"Selected features: {selected_features}")
    
    # Use selected features
    X_selected = X[selected_features]
    
    # Train-test split
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_selected, Y, test_size=0.3, random_state=42, stratify=Y
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )
    
    # Handle class imbalance with SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, Y_train)
    
    # Train final model with reduced parameters for smaller size
    print("Training Random Forest model...")
    rf_final = RandomForestClassifier(
        n_estimators=50,  # Reduced from 200 to 50
        max_depth=10,    # Limited depth to prevent overfitting and reduce size
        max_features='sqrt',  # Use sqrt of features for better performance
        min_samples_split=10,  # Increase to reduce tree complexity
        min_samples_leaf=5,   # Increase to reduce tree complexity
        random_state=42
    )
    
    # Combine train and validation for final training
    X_train_final = np.concatenate([X_train_res, X_val], axis=0)
    y_train_final = np.concatenate([y_train_res, Y_val], axis=0)
    
    rf_final.fit(X_train_final, y_train_final)
    
    # Save model and preprocessing objects with compression
    print("Saving model and preprocessing objects...")
    joblib.dump(rf_final, 'diabetes_model.pkl', compress=3)  # Add compression
    joblib.dump(label_encoders, 'label_encoders.pkl', compress=3)
    joblib.dump(selected_features, 'selected_features.pkl', compress=3)
    
    # Test the model
    y_test_probs = rf_final.predict_proba(X_test.values)[:, 1]
    threshold = 0.89
    y_test_pred = (y_test_probs >= threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(Y_test, y_test_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_test, y_test_pred))
    
    print("Model training completed and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
