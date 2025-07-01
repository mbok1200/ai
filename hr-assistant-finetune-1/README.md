# HR Assistant Fine-Tuning Project

This project aims to fine-tune a model to serve as an HR assistant capable of answering questions related to HR topics, skills, and feedback mechanisms. The model will be trained using a dataset containing various texts relevant to HR practices.

## Project Structure

The project is organized as follows:

```
hr-assistant-finetune
├── data
│   └── all_texts.jsonl          # Training data for the HR assistant
├── src
│   ├── prepare_dataset.py        # Script for data loading and preprocessing
│   ├── train.py                  # Script for training the HR assistant model
│   └── evaluate.py               # Script for evaluating the trained model
├── notebooks
│   └── data_analysis.ipynb       # Jupyter notebook for exploratory data analysis
├── requirements.txt              # Python dependencies for the project
└── README.md                     # Project documentation
```

## Getting Started

### Prerequisites

Make sure you have Python 3.6 or higher installed. You will also need to install the required packages listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd hr-assistant-finetune
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare the Dataset**: Run the `prepare_dataset.py` script to load and preprocess the data.
   ```
   python src/prepare_dataset.py
   ```

2. **Train the Model**: Use the `train.py` script to train the HR assistant model.
   ```
   python src/train.py
   ```

3. **Evaluate the Model**: After training, evaluate the model's performance using the `evaluate.py` script.
   ```
   python src/evaluate.py
   ```

4. **Data Analysis**: Explore the dataset using the Jupyter notebook located in the `notebooks` directory.
   ```
   jupyter notebook notebooks/data_analysis.ipynb
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.