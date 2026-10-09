# Credit Card Fraud Detection

A Machine Learning project that predicts whether a credit card transaction may be fraudulent using a Random Forest Classifier.

## Live Demo

[Click here to try the Credit Card Fraud Detection App](https://credit-card-fraud-detection-pqxjdtpgc5e84red4x22ok.streamlit.app)

## Project Overview

Credit card fraud detection is an imbalanced classification problem. This project uses transaction details and engineered features to estimate the probability of fraud.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Features

- Transaction amount and location inputs
- Transaction time and customer age features
- Customer-to-merchant distance feature
- Estimated fraud probability
- Classification using a configurable decision threshold

## Model Performance

On the held-out test dataset, the model achieved:

- Accuracy: 99.79%
- Precision: 73.84%
- Recall: 69.09%
- F1-score: 71.39%
- ROC-AUC: 98.97%

These metrics reflect evaluation on the test dataset; they do not guarantee real-world performance.

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Disclaimer

This application provides machine-learning estimates for educational purposes. It should not be used as the sole basis for real financial or fraud decisions.
