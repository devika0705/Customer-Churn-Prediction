# Customer Churn Prediction Project

This project predicts customer churn using machine learning. It covers data generation, exploratory data analysis, data scaling, balancing with SMOTE, and model training (Logistic Regression, Random Forest, XGBoost).

## Setup Instructions

1. **Install Dependencies**
   Open your terminal/command prompt, specify the environment you wish to use, and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Synthetic Data**
   Since we aren't using an external CSV, you need to generate it:
   ```bash
   python data_generator.py
   ```
   This creates a `churn_data.csv` file in your repository.

3. **Run the Notebook**
   Launch Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```
   Or open the `.ipynb` file in VS Code. Simply run all cells.

## Models Compared
- Logistic Regression
- Random Forest
- XGBoost

## Highlights
- **Recall prioritization**: Evaluates algorithms based on identifying the most churn candidates correctly.
- **SMOTE**: Implemented to oversample the minority 'Churn' class.
