import json
import random
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def preprocess_data(data):
    # Here you can add any preprocessing steps you need
    # For example, tokenization, normalization, etc.
    processed_data = []
    for entry in data:
        text = entry['text'].strip()
        if text:  # Ensure the text is not empty
            processed_data.append(text)
    return processed_data

def split_data(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

def save_data(train_data, val_data, train_file, val_file):
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item}\n")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(f"{item}\n")

def main():
    data_file = '../data/all_texts.jsonl'
    train_file = '../data/train_data.txt'
    val_file = '../data/val_data.txt'
    
    data = load_data(data_file)
    processed_data = preprocess_data(data)
    train_data, val_data = split_data(processed_data)
    save_data(train_data, val_data, train_file, val_file)

if __name__ == "__main__":
    main()