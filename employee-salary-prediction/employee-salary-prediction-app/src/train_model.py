import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv('employee_data.csv')

# Encode categorical variables
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Education Level'] = le_education.fit_transform(data['Education Level'])
data['Job Title'] = le_job.fit_transform(data['Job Title'])

# Features and target
X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = data['Salary']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'salary_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_education, 'le_education.pkl')
joblib.dump(le_job, 'le_job.pkl')
print('Model and encoders saved successfully.') 