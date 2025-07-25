from src.data.load_data import load_csv, clean_data
from src.models.salary_predictor import SalaryPredictor

def train():
    # Load and preprocess the dataset
    data = load_csv('path/to/dataset.csv')
    cleaned_data = clean_data(data)

    # Initialize the SalaryPredictor
    predictor = SalaryPredictor()

    # Train the model
    predictor.train_model(cleaned_data)

    # Save the trained model (optional)
    predictor.save_model('path/to/save/model.pkl')

if __name__ == "__main__":
    train()