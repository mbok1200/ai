import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def evaluate_model(model, tokenizer, data):
    model.eval()
    predictions, true_labels = [], []

    for item in data:
        inputs = tokenizer(item['text'], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            predictions.append(predicted_class)
            true_labels.append(item['label'])  # Assuming 'label' is the key for true labels

    return predictions, true_labels

def compute_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

def main():
    model_name = "your_model_name"  # Replace with your model name
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    validation_data = load_data('data/all_texts.jsonl')  # Load your validation dataset
    predictions, true_labels = evaluate_model(model, tokenizer, validation_data)
    accuracy, f1 = compute_metrics(predictions, true_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()