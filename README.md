# Credit_card_fraud_detection
## `main.py` - The Core Script

This file contains the complete Python code for the credit card fraud detection system. The script performs the following steps:

1.  **Data Loading:** Loads the `creditcard.csv` dataset using pandas.
2.  **Data Splitting:** Splits the data into training and testing sets.
3.  **Feature Scaling:** Applies `StandardScaler` to normalize the transaction features.
4.  **Imbalance Handling:** Uses `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data to create a balanced set for model training.
5.  **Model Training & Evaluation:**
    *   **Logistic Regression:** A baseline model is trained and evaluated.
    *   **Autoencoder:** An unsupervised neural network is trained on normal data to detect anomalies (fraud).
    *   **XGBoost:** A high-performance gradient boosting model is trained on the balanced data.
6.  **Real-Time Simulation:** A simple loop simulates how the trained XGBoost model would make predictions on a stream of new, incoming transactions.

The script is designed to be run from top to bottom and will print the evaluation results, including confusion matrices and classification reports, for each model.

