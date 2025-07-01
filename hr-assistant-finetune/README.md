# HR Assistant Fine-Tuning Project

This project is designed to fine-tune an HR assistant model that can generate responses based on various HR competencies, feedback mechanisms, and communication skills. The model is trained using a dataset that includes relevant texts in JSON Lines format.

## Project Structure

```
hr-assistant-finetune
├── data
│   └── all_texts.jsonl         # Training data in JSON Lines format
├── src
│   ├── main.py                  # Entry point of the application
│   ├── model.py                 # Defines the HR assistant model
│   └── utils.py                 # Utility functions for data processing
├── notebooks
│   └── data_analysis.ipynb      # Jupyter notebook for data analysis
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd hr-assistant-finetune
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the HR assistant fine-tuning process, execute the following command:

```
python src/main.py
```

This will initialize the model, load the training data from `data/all_texts.jsonl`, and start the fine-tuning process.

## Functionality

The HR assistant model is designed to assist with:

- Generating feedback based on employee performance.
- Providing insights into communication skills and competencies.
- Offering guidance on effective HR practices.

## Data Analysis

The `notebooks/data_analysis.ipynb` file contains exploratory data analysis of the training dataset. It includes visualizations and insights that can help in understanding the data better and improving the model's performance.

## Contributions

Contributions to the project are welcome. Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.