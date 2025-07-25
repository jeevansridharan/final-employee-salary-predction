import pandas as pd

# Load the dataset
data = pd.read_csv('employee_data.csv')

# Show the first few rows
print(data.head())

# Show basic info
print(data.info())