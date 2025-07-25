# Employee Salary Prediction Project

This project aims to predict employee salaries using machine learning algorithms. It includes various components for data loading, preprocessing, model training, and prediction.

## Project Structure

```
employee-salary-prediction
├── src
│   ├── data
│   │   └── load_data.py       # Functions to load and preprocess the dataset
│   ├── models
│   │   └── salary_predictor.py  # Class for salary prediction model
│   ├── utils
│   │   └── preprocess.py        # Utility functions for data preprocessing
│   ├── train.py                 # Script to train the model
│   ├── predict.py               # Script to make predictions
│   └── requirements.txt         # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd employee-salary-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r src/requirements.txt
   ```

## Usage

### Training the Model

To train the salary prediction model, run the following command:
```
python src/train.py
```

### Making Predictions

To make salary predictions using the trained model, use:
```
python src/predict.py
```

## File Descriptions

- **load_data.py**: Contains functions to load and clean the dataset.
- **salary_predictor.py**: Defines the `SalaryPredictor` class with methods for training and predicting salaries.
- **preprocess.py**: Provides utility functions for normalizing and encoding data.
- **train.py**: Responsible for training the model using the dataset.
- **predict.py**: Handles the prediction process with the trained model.
- **requirements.txt**: Lists all necessary libraries for the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.