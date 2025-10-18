import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    .risk-low {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('diabetes_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        selected_features = joblib.load('selected_features.pkl')
        return model, label_encoders, selected_features
    except FileNotFoundError:
        st.error("Model files not found. Please run 'python train_model.py' first.")
        return None, None, None

def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, label_encoders, selected_features):
    """Preprocess user input for prediction"""
    # Create a dataframe with the input data
    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension else 0],
        'heart_disease': [1 if heart_disease else 0],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    }
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables using the appropriate encoders
    df['gender'] = label_encoders['gender'].transform(df['gender'])
    df['smoking_history'] = label_encoders['smoking_history'].transform(df['smoking_history'])
    
    # Select only the features used in training
    df_selected = df[selected_features]
    
    # Convert to numpy array to avoid feature name warnings
    return df_selected.values

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, label_encoders, selected_features = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📋 Patient Information")
    
    # Input fields
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Male", "Female", "Other"]
    )
    
    age = st.sidebar.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        help="Patient's age in years"
    )
    
    hypertension = st.sidebar.checkbox(
        "Hypertension",
        help="High blood pressure condition"
    )
    
    heart_disease = st.sidebar.checkbox(
        "Heart Disease",
        help="History of heart disease"
    )
    
    smoking_history = st.sidebar.selectbox(
        "Smoking History",
        options=["never", "former", "current", "not current", "ever", "no info"]
    )
    
    bmi = st.sidebar.slider(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (kg/m²)"
    )
    
    hba1c_level = st.sidebar.slider(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Hemoglobin A1c level (%)"
    )
    
    blood_glucose_level = st.sidebar.slider(
        "Blood Glucose Level",
        min_value=50,
        max_value=300,
        value=100,
        help="Blood glucose level (mg/dL)"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Diabetes Risk", type="primary"):
        # Preprocess input
        input_data = preprocess_input(
            gender, age, hypertension, heart_disease, 
            smoking_history, bmi, hba1c_level, blood_glucose_level,
            label_encoders, selected_features
        )
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Prediction Results")
            
            # Risk level
            risk_level = "High Risk" if probability >= 0.5 else "Low Risk"
            risk_color = "⚠️" if probability >= 0.5 else "✅"
            
            st.markdown(f"""
            <div class="prediction-box {'risk-high' if probability >= 0.5 else 'risk-low'}">
                <h3>{risk_color} {risk_level}</h3>
                <p><strong>Probability:</strong> {probability:.2%}</p>
                <p><strong>Confidence:</strong> {'High' if abs(probability - 0.5) > 0.3 else 'Medium'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Risk Factors Analysis")
            
            # Create a simple risk factors chart
            risk_factors = {
                'Age': min(age/100, 1.0),
                'BMI': min(bmi/50, 1.0),
                'HbA1c': min(hba1c_level/15, 1.0),
                'Glucose': min(blood_glucose_level/300, 1.0)
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=['#ff6b6b', '#00b894', '#667eea', '#fdcb6e'],
                    marker_line=dict(color='white', width=2)
                )
            ])
            
            fig.update_layout(
                title="Risk Factor Levels",
                yaxis_title="Normalized Risk Level",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.markdown("### 🔍 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Age Risk",
                value="High" if age > 65 else "Medium" if age > 45 else "Low",
                delta=f"{age} years"
            )
        
        with col2:
            st.metric(
                label="BMI Risk",
                value="High" if bmi > 30 else "Medium" if bmi > 25 else "Low",
                delta=f"{bmi:.1f}"
            )
        
        with col3:
            st.metric(
                label="HbA1c Risk",
                value="High" if hba1c_level > 6.5 else "Medium" if hba1c_level > 5.7 else "Low",
                delta=f"{hba1c_level:.1f}%"
            )
        
        with col4:
            st.metric(
                label="Glucose Risk",
                value="High" if blood_glucose_level > 140 else "Medium" if blood_glucose_level > 100 else "Low",
                delta=f"{blood_glucose_level} mg/dL"
            )
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if probability >= 0.5:
            st.warning("""
            **High Risk Detected!** 
            
            Please consult with a healthcare professional immediately. Consider:
            - Regular blood glucose monitoring
            - Lifestyle modifications (diet and exercise)
            - Regular medical check-ups
            - Medication consultation if needed
            """)
        else:
            st.success("""
            **Low Risk** 
            
            Continue maintaining a healthy lifestyle:
            - Regular exercise
            - Balanced diet
            - Regular health check-ups
            - Monitor blood glucose levels periodically
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        " Diabetes Risk Prediction System"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
