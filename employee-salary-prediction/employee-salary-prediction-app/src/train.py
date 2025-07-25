import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv('data/employees.csv')  # Updated path

# Features and target
X = data.drop('salary', axis=1)
y = data['salary']

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/salary_model.pkl')  # Updated path

print("Model trained and saved!")