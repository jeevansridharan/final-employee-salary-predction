def load_csv(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Example cleaning steps
    data = data.dropna()  # Remove missing values
    data = data.reset_index(drop=True)  # Reset index after dropping
    return data

def load_and_preprocess_data(file_path):
    data = load_csv(file_path)
    cleaned_data = clean_data(data)
    return cleaned_data