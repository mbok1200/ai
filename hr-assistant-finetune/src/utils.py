def load_data(file_path):
    import json

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def preprocess_text(text):
    import re

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def save_model(model, file_path):
    import joblib

    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib

    return joblib.load(file_path)