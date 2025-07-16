# Ollama ML Test Project

This project is designed to test and evaluate models using the Ollama API. It provides a structured approach to load, preprocess data, and interact with the Ollama client for generating responses.

## Project Structure

```
ollama-ml-test-project
├── src
│   ├── main.py                # Entry point of the application
│   ├── models
│   │   ├── __init__.py        # Models package initialization
│   │   └── ollama_client.py    # OllamaClient class for API interaction
│   ├── data
│   │   ├── __init__.py        # Data package initialization
│   │   └── preprocessor.py     # DataPreprocessor class for data handling
│   ├── tests
│   │   ├── __init__.py        # Tests package initialization
│   │   └── test_models.py      # Unit tests for the OllamaClient
│   └── utils
│       ├── __init__.py        # Utils package initialization
│       └── config.py          # Configuration loading utility
├── requirements.txt            # Project dependencies
├── config.yaml                 # Configuration settings for the project
└── README.md                   # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ollama-ml-test-project.git
   cd ollama-ml-test-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your settings in `config.yaml`:
   - Add your API keys and model parameters.

## Usage

To run the application, execute the following command:
```
python src/main.py
```

## Testing

To run the unit tests, use:
```
pytest src/tests/test_models.py
```

## Purpose

The Ollama ML Test Project aims to provide a framework for testing machine learning models using the Ollama API. It facilitates easy data preprocessing, model interaction, and response generation, making it a valuable tool for developers and researchers in the field of machine learning.