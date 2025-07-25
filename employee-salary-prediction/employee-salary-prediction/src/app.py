from flask import Flask, request, jsonify, send_from_directory
from joblib import load
import pandas as pd
import os
from models.salary_predictor import SalaryPredictor
import numpy as np

# Categorical encoding mappings based on training data
GENDER_MAP = {'Male': 1, 'Female': 0}
EDUCATION_MAP = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
JOB_TITLE_MAP = {
    'Software Engineer': 6,
    'Data Scientist': 1,
    'Web Developer': 7,
    'Research Scientist': 5,
    'Project Manager': 4,
    'Intern': 3,
    'Senior Developer': 8,
    'HR Manager': 2,
    'Data Analyst': 0,
    'CTO': 9
}

CATEGORICAL_COLUMNS = ['gender', 'education_level', 'job_title']

NUMERIC_COLUMNS = ['age', 'years_of_experience', 'performance_score']

ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_ORDER = ['age', 'gender', 'education_level', 'job_title', 'years_of_experience', 'performance_score']

def preprocess_input(df):
    df = df.copy()
    # Map categorical columns
    df['gender'] = df['gender'].map(GENDER_MAP).fillna(-1).astype(int)
    df['education_level'] = df['education_level'].map(EDUCATION_MAP).fillna(-1).astype(int)
    df['job_title'] = df['job_title'].map(JOB_TITLE_MAP).fillna(-1).astype(int)
    # Ensure correct column order
    return df[FEATURE_ORDER]

app = Flask(__name__)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../employee-salary-prediction-app/models/salary_model.pkl'))

def load_model():
    return load(MODEL_PATH)

@app.route('/')
def index():
    return send_from_directory('.', 'frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()
        # Handle JSON input (single prediction)
        if request.is_json:
            input_data = pd.DataFrame([request.get_json()])
        # Handle CSV file upload (batch prediction)
        elif 'file' in request.files:
            file = request.files['file']
            input_data = pd.read_csv(file)
        else:
            return jsonify({'error': 'No valid input provided. Send JSON or upload a CSV file.'}), 400
        # Preprocess input
        input_data = preprocess_input(input_data)
        # Predict
        predictions = model.predict(input_data)
        return jsonify({'predictions': np.round(predictions, 2).tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/avg_salary')
def avg_salary():
    try:
        df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../employee-salary-prediction-app/data/employees.csv')))
        avg = float(df['salary'].mean())
        return jsonify({'avg_salary': round(avg, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 