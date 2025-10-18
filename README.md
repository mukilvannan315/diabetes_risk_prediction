# Diabetes Risk Prediction 🩺

A comprehensive machine learning project for predicting diabetes risk using multiple health indicators. This project implements various classification algorithms and provides an interactive Streamlit web application for real-time predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Diabetes is a chronic disease that affects millions of people worldwide. Early prediction and detection can help in better management and prevention of complications. This project uses machine learning algorithms to predict the risk of diabetes based on various health parameters.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Multiple machine learning model implementations
- Model evaluation and comparison
- Interactive web application for predictions
- Model serialization for deployment

## 📊 Dataset

The project uses the **Diabetes Prediction Dataset** (`diabetes_prediction_dataset.csv`) which contains health information of individuals including:

- **Sample Size**: Comprehensive dataset with multiple health indicators
- **Target Variable**: Diabetes status (0 = No Diabetes, 1 = Diabetes)
- **Features**: 8 key health indicators

### Dataset Features:
1. **Age**: Age of the individual
2. **Gender**: Gender (Male/Female)
3. **BMI**: Body Mass Index
4. **Hypertension**: Hypertension status (0 = No, 1 = Yes)
5. **Heart Disease**: Heart disease status (0 = No, 1 = Yes)
6. **Smoking History**: Smoking history category
7. **HbA1c Level**: Hemoglobin A1c level (average blood sugar)
8. **Blood Glucose Level**: Blood glucose concentration

## 🎨 Features

### Data Preprocessing
- Handling missing values
- Encoding categorical variables (Label Encoding)
- Feature scaling and normalization
- Feature selection using statistical methods
- Train-test split for model validation

### Exploratory Data Analysis
- Distribution analysis of features
- Correlation analysis
- Class imbalance investigation
- Visualization of key relationships

## 🤖 Machine Learning Models

The project implements and compares multiple classification algorithms:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Gradient Boosting Classifier**
7. **XGBoost Classifier**

### Model Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Cross-validation scores

## 📁 Project Structure

```
diabetes_risk_prediction/
│
├── main.ipynb                          # Main Jupyter notebook with complete analysis
├── train_model.py                      # Script to train and save the model
├── app.py                              # Streamlit web application
├── run_app.py                          # Script to run the Streamlit app
├── test_prediction.py                  # Testing script for predictions
├── setup.py                            # Setup and installation script
├── status_check.py                     # Status checking utility
│
├── diabetes_prediction_dataset.csv     # Dataset
├── diabetes_model.pkl                  # Trained model (serialized)
├── label_encoders.pkl                  # Label encoders for categorical features
├── selected_features.pkl               # Selected feature names
│
└── README.md                           # Project documentation
```

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/mukilvannan315/diabetes_risk_prediction.git
cd diabetes_risk_prediction
```

### Step 2: Install Dependencies

#### Option 1: Using setup.py
```bash
python setup.py
```

This will automatically install all required packages:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- streamlit
- pickle

#### Option 2: Manual Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

## 🚀 Usage

### 1. Training the Model

To train the model from scratch:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the machine learning model
- Save the trained model as `diabetes_model.pkl`
- Save label encoders and feature names

### 2. Running the Web Application

To launch the interactive Streamlit web application:

```bash
python run_app.py
```

Or directly:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 3. Making Predictions

To test predictions programmatically:

```bash
python test_prediction.py
```

### 4. Jupyter Notebook Analysis

To explore the complete analysis:

```bash
jupyter notebook main.ipynb
```

## 📈 Model Performance

The models are evaluated using various metrics. The best performing model is selected based on:
- Cross-validation accuracy
- Generalization ability
- Precision-Recall trade-off

*Note: Detailed performance metrics can be found in the `main.ipynb` notebook.*

## 💻 Web Application

The Streamlit web application provides an intuitive interface for diabetes risk prediction:

### Features:
- **User-friendly input forms** for health parameters
- **Real-time predictions** using the trained model
- **Risk assessment** with probability scores
- **Interactive interface** with clear visualizations
- **Input validation** to ensure data quality

### How to Use the Web App:
1. Launch the application using `python run_app.py`
2. Enter the required health information:
   - Age
   - Gender
   - BMI
   - Hypertension status
   - Heart disease status
   - Smoking history
   - HbA1c level
   - Blood glucose level
3. Click on "Predict" to get the diabetes risk assessment
4. View the prediction result and probability

## 🛠️ Technologies Used

### Programming Language
- **Python 3.x**

### Data Processing & Analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools

### Machine Learning
- **Scikit-learn**: Classical ML algorithms
- **XGBoost**: Gradient boosting framework

### Data Visualization
- **Matplotlib**: Plotting library
- **Seaborn**: Statistical data visualization

### Web Application
- **Streamlit**: Web application framework

### Model Deployment
- **Pickle**: Model serialization

## 📊 Key Insights

From the exploratory data analysis:
- Certain health indicators show strong correlation with diabetes risk
- Age, BMI, HbA1c level, and blood glucose level are significant predictors
- The dataset may exhibit class imbalance which is handled appropriately
- Ensemble methods generally perform better than individual classifiers

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

### Areas for Contribution:
- Improving model performance
- Adding more visualization features
- Enhancing the web application UI/UX
- Adding more machine learning algorithms
- Improving documentation
- Adding unit tests

## 📝 Future Enhancements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add feature importance visualization
- [ ] Implement SHAP values for model interpretability
- [ ] Add data augmentation techniques for imbalanced data
- [ ] Deploy the application to cloud platforms (Heroku, AWS, etc.)
- [ ] Add user authentication and history tracking
- [ ] Implement batch prediction functionality
- [ ] Add API endpoints for integration

## 👤 Author

**Mukil Vannan**
- GitHub: [@mukilvannan315](https://github.com/mukilvannan315)

## 🙏 Acknowledgments

- Dataset source: Diabetes Prediction Dataset
- Inspired by the need for early diabetes detection and prevention
- Thanks to the open-source community for the amazing tools and libraries

