import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("The_Cancer_data_1500_V2.copy.csv")
    
    # Handling missing values
    data['BMI'].fillna(data['BMI'].median(), inplace=True)
    data['PhysicalActivity'].fillna(data['PhysicalActivity'].mode()[0], inplace=True)
    data['AlcoholIntake'].fillna(data['AlcoholIntake'].mean(), inplace=True)
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in ["Smoking", "GeneticRisk", "CancerHistory", "Gender"]:
        data[col] = le.fit_transform(data[col])
        
    data['Diagnosis'] = data['Diagnosis'].map({"No Cancer": 0, "Cancer": 1})
    data.dropna(inplace=True)
    return data

data = load_data()

# Features and target
X = data.drop(columns="Diagnosis")
y = data["Diagnosis"]

# Train model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Streamlit UI
st.title("Cancer Prediction App")
st.write("Enter the health information below to predict if the person may have cancer.")

# User input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
smoking = st.selectbox("Smoking", ["No", "Yes"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
genetic_risk = st.selectbox("Genetic Risk", ["None", "Mild", "Moderate", "High"])
cancer_history = st.selectbox("Cancer History", ["No", "Yes"])
alcohol = st.number_input("Alcohol Intake", min_value=0.0, max_value=20.0, value=5.0)
physical = st.number_input("Physical Activity", min_value=0.0, max_value=10.0, value=3.0)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode inputs
smoking_val = 1 if smoking == "Yes" else 0
genetic_val = {"None": 0, "Mild": 1, "Moderate": 2, "High": 3}[genetic_risk]
cancer_val = 1 if cancer_history == "Yes" else 0
gender_val = 1 if gender == "Male" else 0

# Predict
if st.button("Predict"):
    input_data = np.array([[age, smoking_val, bmi, cancer_val, genetic_val, alcohol, physical, gender_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("The person is not likely to have Cancer.")
    else:
        st.error("The person is likely to have Cancer.")
