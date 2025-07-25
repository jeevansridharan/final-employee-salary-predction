from joblib import load
import pandas as pd
from models.salary_predictor import SalaryPredictor

def load_input_data(file_path):
    # Load new input data for prediction
    return pd.read_csv(file_path)

def make_prediction(model_path, input_data_path):
    # Load the trained model
    model = load(model_path)
    
    # Load the input data
    input_data = load_input_data(input_data_path)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions

if __name__ == "__main__":
    model_path = 'path/to/trained_model.joblib'  # Update with the actual model path
    input_data_path = 'path/to/input_data.csv'   # Update with the actual input data path
    
    predictions = make_prediction(model_path, input_data_path)
    print(predictions)