# Employee Salary Prediction App

This project is a web application for predicting employee salaries using machine learning algorithms. The application utilizes a dataset containing various employee details and employs an XGBoost Regressor for salary prediction.

## Project Structure

```
employee-salary-prediction-app
├── data
│   └── employees.csv          # Dataset containing employee details
├── src
│   ├── app.py                 # Main entry point of the Streamlit application
│   ├── model.py               # Implementation of the machine learning model
│   ├── preprocess.py          # Data preprocessing tasks
│   └── utils.py               # Utility functions for the application
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd employee-salary-prediction-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Features

- User-friendly interface for salary prediction
- Input fields for employee details such as age, gender, education level, job title, and years of experience
- Display of predicted salary along with relevant information

## Model Information

The application uses an XGBoost Regressor for predicting salaries based on the input features. The model is trained on a dataset that includes various employee attributes.

## License

This project is licensed under the MIT License.