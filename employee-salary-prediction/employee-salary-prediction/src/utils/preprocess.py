def normalize_data(data):
    """Normalize the numerical features in the dataset."""
    return (data - data.mean()) / data.std()

def encode_categorical(data, categorical_columns):
    """Encode categorical features using one-hot encoding."""
    return pd.get_dummies(data, columns=categorical_columns, drop_first=True)