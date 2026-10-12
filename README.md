
# Customer Churn Prediction

## Project Overview

This project uses machine learning to predict whether a telecom customer is likely to leave a service.

A Random Forest classifier is trained using customer demographics, account information, contract details, internet services, and billing data. A Streamlit web application allows users to enter customer information and view churn predictions.

## Objectives

- Clean and preprocess customer data.
- Encode categorical variables using One-Hot Encoding.
- Train and tune a Random Forest classifier.
- Evaluate performance using precision, recall, F1-score, and a confusion matrix.
- Explain model behaviour using feature importance and SHAP.
- Build an interactive web application for predictions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- SHAP
- Joblib
- Streamlit

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Clean missing and invalid values.
3. Separate features and target.
4. Apply One-Hot Encoding to categorical features.
5. Train a Random Forest classifier.
6. Tune hyperparameters and evaluate performance.
7. Analyse feature importance and model explanations.
8. Save the trained pipeline.
9. Build an interactive Streamlit application.

## Model Performance

The final model achieved the following results on the test split:

| Metric | Score |
|---|---:|
| Accuracy | 75.37% |
| Precision (Churn = Yes) | 52.42% |
| Recall (Churn = Yes) | 78.34% |
| F1-score (Churn = Yes) | 62.81% |

The model identifies many customers who churned, but its precision indicates that some customers predicted to churn did not actually churn.

These results are specific to the dataset and evaluation split used in this project. They do not guarantee the same performance on new customer data.

## Streamlit Application

The web application allows users to:

- Enter customer information.
- Obtain a predicted churn class.
- View the estimated churn probability.
- Review predictions in a session-based history table.

## Project Structure

```text
customer_churn_random_forest/
├── app.py
├── churn_model.pkl
├── churn_prediction.py
├── README.md
└── requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Start the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Estimated churn probabilities may require calibration before being interpreted as real-world probabilities.
- The model may make false-positive and false-negative predictions.
- Prediction history is temporary and is not stored in a permanent database.
- Model performance should be re-evaluated on a genuinely untouched test set before making strong claims about generalisation.

## Future Improvements

- Improve input validation and user experience.
- Evaluate probability calibration.
- Add customer retention recommendations.
- Deploy the application online.
- Add automated tests and reproducible dependency versions.

## Disclaimer

This project is for educational and portfolio purposes. Predictions should not be used as the sole basis for real customer decisions.