# predict.py

import pandas as pd
import joblib
import os

# Load the trained model (make sure the path is correct)
model_path = os.path.join("models", "salary_model.pkl")
le_gender_path = os.path.join(os.path.dirname(__file__), "le_gender.pkl")
le_education_path = os.path.join(os.path.dirname(__file__), "le_education.pkl")
le_job_path = os.path.join(os.path.dirname(__file__), "le_job.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")
if not os.path.exists(le_gender_path):
    raise FileNotFoundError(f"Gender encoder not found at: {le_gender_path}")
if not os.path.exists(le_education_path):
    raise FileNotFoundError(f"Education encoder not found at: {le_education_path}")
if not os.path.exists(le_job_path):
    raise FileNotFoundError(f"Job encoder not found at: {le_job_path}")

model = joblib.load(model_path)
le_gender = joblib.load(le_gender_path)
le_education = joblib.load(le_education_path)
le_job = joblib.load(le_job_path)

# Get valid options from encoders
valid_genders = list(le_gender.classes_)
valid_educations = list(le_education.classes_)
valid_jobs = list(le_job.classes_)

# Helper to get and validate input

def get_input(prompt, valid_options=None, cast_type=str):
    while True:
        if valid_options:
            print(f"  Options: {valid_options}")
        value = input(prompt)
        try:
            value_cast = cast_type(value)
        except Exception:
            print(f"Invalid input type. Please enter a {cast_type.__name__}.")
            continue
        if valid_options and value_cast not in valid_options:
            print(f"Invalid option. Please choose from the listed options.")
            continue
        return value_cast

print("\n--- Employee Salary Prediction ---\n")
age = get_input("Enter Age: ", cast_type=int)
gender = get_input("Enter Gender: ", valid_options=valid_genders)
education_level = get_input("Enter Education Level: ", valid_options=valid_educations)
job_title = get_input("Enter Job Title: ", valid_options=valid_jobs)
years_of_experience = get_input("Enter Years of Experience: ", cast_type=int)
performance_score = get_input("Enter Performance Score: ", cast_type=int)

print("\n--- Employee Details Entered ---")
print(f"Age: {age}")
print(f"Gender: {gender}")
print(f"Education Level: {education_level}")
print(f"Job Title: {job_title}")
print(f"Years of Experience: {years_of_experience}")
print(f"Performance Score: {performance_score}")
print("-------------------------------\n")

new_employee = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'education_level': education_level,
    'job_title': job_title,
    'years_of_experience': years_of_experience,
    'performance_score': performance_score
}])

# ⚠️ IMPORTANT: Encoding should match the training phase.
# If you used LabelEncoder or OneHotEncoder, load and apply them here.
# For now, we’ll assume a pipeline was used, so no manual encoding needed.

# Encode categorical features
try:
    print("Valid options:")
    print("  Gender:", le_gender.classes_)
    print("  Education Level:", le_education.classes_)
    print("  Job Title:", le_job.classes_)

    # Check if input values are valid
    if new_employee["gender"].iloc[0] not in le_gender.classes_:
        print(f"❌ Invalid gender: {new_employee['gender'].iloc[0]}. Allowed: {le_gender.classes_}")
        exit(1)
    if new_employee["education_level"].iloc[0] not in le_education.classes_:
        print(f"❌ Invalid education_level: {new_employee['education_level'].iloc[0]}. Allowed: {le_education.classes_}")
        exit(1)
    if new_employee["job_title"].iloc[0] not in le_job.classes_:
        print(f"❌ Invalid job_title: {new_employee['job_title'].iloc[0]}. Allowed: {le_job.classes_}")
        exit(1)

    new_employee["gender"] = le_gender.transform(new_employee["gender"])
    new_employee["education_level"] = le_education.transform(new_employee["education_level"])
    new_employee["job_title"] = le_job.transform(new_employee["job_title"])
    print("Encoded employee data:")
    print(new_employee)
except Exception as e:
    print("❌ Error while encoding categorical features:", e)
    import traceback
    traceback.print_exc()
    exit(1)

# Predict salary
try:
    predicted_salary = model.predict(new_employee)
    print("--- Prediction Result ---")
    print(f"✅ Predicted Salary: ₹{predicted_salary[0]:,.2f}")
    print("------------------------\n")
except Exception as e:
    print("❌ Error while predicting salary:", e)
    import traceback
    traceback.print_exc()

# Calculate and print average salary from employee_data.csv
try:
    csv_path = os.path.join(os.path.dirname(__file__), 'employee_data.csv')
    df = pd.read_csv(csv_path)
    avg_salary = df['Salary'].mean()
    print("--- Average Salary Data ---")
    print(f"📊 Average Salary in employee_data.csv: ₹{avg_salary:,.2f}")
    print("--------------------------\n")
except Exception as e:
    print("❌ Error while calculating average salary:", e)
    import traceback
    traceback.print_exc()
