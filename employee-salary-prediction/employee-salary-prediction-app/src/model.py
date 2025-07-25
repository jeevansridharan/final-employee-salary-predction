from xgboost import XGBRegressor
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

class SalaryPredictor:
    def __init__(self):
        self.model = XGBRegressor()
    
    def train(self, data_path):
        data = pd.read_csv(data_path)
        X = data.drop('salary', axis=1)
        y = data['salary']
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(self.model, 'salary_model.pkl')
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, features):
        prediction = self.model.predict([features])
        scaled_prediction = prediction[0] * 0.8  # Reduce by 20%
        return scaled_prediction