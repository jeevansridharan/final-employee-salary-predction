from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Encode categorical variables
    categorical_features = ['gender', 'education_level', 'job_title']
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    
    # Scale numerical features
    numerical_features = ['age', 'years_of_experience']
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(data[numerical_features])
    
    # Combine processed features
    processed_data = pd.concat([pd.DataFrame(scaled_numerical), 
                                 pd.DataFrame(encoded_categorical)], axis=1)
    
    return processed_data, encoder, scaler

def split_features_target(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y