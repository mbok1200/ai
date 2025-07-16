from models.ollama_client import OllamaClient
from data.preprocessor import DataPreprocessor
from utils.config import load_config

def main():
    # Load configuration
    config = load_config()

    # Initialize the Ollama client
    ollama_client = OllamaClient()
    ollama_client.initialize(config['api_key'])

    # Load and preprocess data
    data_preprocessor = DataPreprocessor()
    raw_data = data_preprocessor.load_data(config['data_path'])
    processed_data = data_preprocessor.preprocess_data(raw_data)

    # Generate response using the model
    response = ollama_client.generate_response(processed_data)
    print("Model Response:", response)

if __name__ == "__main__":
    main()