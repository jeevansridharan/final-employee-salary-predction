import streamlit as st
import pandas as pd
import joblib
import os

# Get the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the dataset for average salary calculation
employee_data = pd.read_csv(os.path.join(BASE_DIR, 'employee_data.csv'))

# Load model and encoders using robust, permanent paths
model = joblib.load(os.path.join(BASE_DIR, '../models/salary_model.pkl'))
le_gender = joblib.load(os.path.join(BASE_DIR, 'le_gender.pkl'))
le_education = joblib.load(os.path.join(BASE_DIR, 'le_education.pkl'))
le_job = joblib.load(os.path.join(BASE_DIR, 'le_job.pkl'))

genders = list(le_gender.classes_)
educations = list(le_education.classes_)
jobs = list(le_job.classes_)

st.title("Employee Salary Prediction")

# User input fields
age = st.number_input("Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("Gender", options=genders)
education = st.selectbox("Education Level", options=educations)
job_title = st.selectbox("Job Title", options=jobs)
years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
performance_score = st.number_input("Performance Score", min_value=1, max_value=10, value=5)

if st.button("Predict Salary"):
    # Encode user input
    gender_enc = le_gender.transform([gender])[0]
    education_enc = le_education.transform([education])[0]
    job_enc = le_job.transform([job_title])[0]
    # Prepare input for prediction (columns must match model)
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender_enc,
        'education_level': education_enc,
        'job_title': job_enc,
        'years_of_experience': years_exp,
        'performance_score': performance_score
    }])
    prediction = model.predict(input_df)
    avg_salary = employee_data['Salary'].mean()
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
    st.info(f"Average Salary in Dataset: ${avg_salary:,.2f}")