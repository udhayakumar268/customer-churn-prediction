# ============================================================
# CUSTOMER CHURN PREDICTION
# Random Forest + Feature Importance + Permutation Importance
# + SHAP Explainability
# ============================================================


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import(accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer,precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

# 2. LOAD AND CLEAN THE DATA

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nDataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Convert TotalCharges to numeric.
# Invalid/blank values become NaN.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# The blank TotalCharges records in this dataset correspond
# to customers with zero tenure. Fill those charges with zero.

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing values after cleaning:", df.isnull().sum().sum())

# 3. SEPARATE FEATURES AND TARGET

X = df.drop("Churn", axis=1)
y = df["Churn"]
X = X.drop("customerID", axis=1)

print("\nTarget distribution:")
print(y.value_counts())

# 4. IDENTIFY NUMERIC AND CATEGORICAL FEATURES

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical features:", len(categorical_columns))
print("Numeric features:", len(numerical_columns))

# 5. PREPROCESSING

preprocessor = ColumnTransformer(transformers=[("categorical", OneHotEncoder(handle_unknown= "ignore"), categorical_columns)], remainder="passthrough")

# 6. SPLIT INTO TRAINING, VALIDATION AND TEST DATA

# First, separate 20% for the final test
X_remaining, X_test_final, y_remaining, y_test_final = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the remaining 80% into training and validation
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.25,
    random_state=42,
    stratify=y_remaining
)

print("\nDataset split:")
print("Training rows:", len(X_train_new))
print("Validation rows:", len(X_val))
print("Final test rows:", len(X_test_final))

# 7. BUILD AND TRAIN THE VALIDATION MODEL

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Train using training data only
rf_model.fit(X_train_new, y_train_new)

print("\nValidation model training completed!")

# 8. COMPARE CLASSIFICATION THRESHOLDS

# Get churn probabilities on validation data
val_probability = rf_model.predict_proba(X_val)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

print("\nVALIDATION THRESHOLD COMPARISON")

for threshold in thresholds:
    val_prediction = (
        rf_model.predict_proba(X_val)[:, 1] >= threshold
    )
    val_prediction = [
        "Yes" if p else "No" for p in val_prediction
    ]

    precision = precision_score(
        y_val, val_prediction, pos_label="Yes"
    )
    recall = recall_score(
        y_val, val_prediction, pos_label="Yes"
    )
    f1 = f1_score(
        y_val, val_prediction, pos_label="Yes"
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# Keep the threshold selected from validation.
CHOSEN_THRESHOLD = 0.50

# 9. FINAL MODEL: TRAIN ON TRAINING + VALIDATION DATA

X_train_full = pd.concat([X_train_new, X_val])
y_train_full = pd.concat([y_train_new, y_val])

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X_train_full, y_train_full)

print("Final model training completed!")

# 10. FINAL TEST EVALUATION

test_probability = final_model.predict_proba(X_test_final)[:, 1]

test_prediction = [
    "Yes" if probability >= 0.50 else "No"
    for probability in test_probability
]
print("\nFINAL TEST RESULTS")
print("Threshold:", CHOSEN_THRESHOLD)
print("Accuracy:", accuracy_score(y_test_final, test_prediction))
print("Precision:", precision_score(
    y_test_final, test_prediction, pos_label="Yes"
))
print("Recall:", recall_score(
    y_test_final, test_prediction, pos_label="Yes"
))
print("F1-score:", f1_score(
    y_test_final, test_prediction, pos_label="Yes"
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test_final, test_prediction, labels=["No", "Yes"]
))

print("\nClassification Report:")
print(classification_report(y_test_final, test_prediction))

# 11. RANDOM FOREST FEATURE IMPORTANCE

# Get the trained Random Forest from the pipeline
rf = final_model.named_steps["classifier"]

# Get the feature names created by preprocessing
feature_names = final_model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get the importance score of each feature
importance = rf.feature_importances_

# Create a table
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort from most important to least important
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df.head(15))
top_10 = feature_importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.xlabel("Feature Importance")
plt.title("Top 10 Features Influencing Churn Predictions")
plt.tight_layout()
plt.show()

# 12. PERMUTATION IMPORTANCE
# Use the validation model on validation data because that
# model was not trained on X_val.

f1_scorer = make_scorer(f1_score, pos_label="Yes")

perm_result = permutation_importance(
    final_model,
    X_val,
    y_val,
    scoring=f1_scorer,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": perm_result.importances_mean
})

permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 PERMUTATION IMPORTANCE")
print(permutation_df.head(10).to_string(index=False))

top_perm = permutation_df.head(10).sort_values("Importance")

plt.figure(figsize=(10, 6))
plt.barh(top_perm["Feature"], top_perm["Importance"])
plt.xlabel("Mean decrease in validation F1-score")
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()

# 13. SHAP: EXPLAIN ONE TEST CUSTOMER

# Select one customer from the validation dataset
customer = X_val.iloc[[0]]

# Get the actual outcome
actual_outcome = y_val.iloc[0]

# Predict the probability of churn
churn_probability = final_model.predict_proba(customer)[0, 1]

# Convert probability into a prediction using threshold 0.50
prediction = "Yes" if churn_probability >= 0.50 else "No"

print("Customer details:")
print(customer.T)

print("\nActual outcome:", actual_outcome)
print(f"Churn probability: {churn_probability:.2%}")
print("Predicted outcome:", prediction)

#  Get the trained Random Forest from your pipeline
rf = final_model.named_steps["classifier"]

# Transform the customer data using your fitted preprocessor
preprocessor = final_model.named_steps["preprocessor"]

customer = X_val.iloc[[0]]
customer_transformed = preprocessor.transform(customer)

# Get the feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Create a SHAP explainer for the Random Forest
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values for this customer
shap_values = explainer.shap_values(customer_transformed)

print("SHAP calculation completed!")
print("Transformed customer shape:", customer_transformed.shape)
print("Feature names:", len(feature_names))


# Get SHAP values for the "Yes" (churn) class
if isinstance(shap_values, list):
    customer_shap = shap_values[1][0]
elif len(shap_values.shape) == 3:
    customer_shap = shap_values[0, :, 1]
else:
    customer_shap = shap_values[0]

# Convert the transformed customer into a flat numeric array
customer_values = np.asarray(customer_transformed).reshape(-1)

# Build a table of feature contributions
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "CustomerValue": customer_values,
    "SHAPValue": customer_shap
})

# Show the largest positive and negative contributions
shap_df["AbsoluteSHAP"] = shap_df["SHAPValue"].abs()
shap_df = shap_df.sort_values("AbsoluteSHAP", ascending=False)

print(shap_df.head(15).to_string(index=False))

# Plot the 10 largest contributions for this customer.
top_shap = shap_df.head(10).sort_values("SHAPValue")

plt.figure(figsize=(10, 6))
plt.barh(top_shap["Feature"], top_shap["SHAPValue"])
plt.axvline(x=0, linewidth=1)
plt.xlabel("SHAP value for churn class")
plt.title("SHAP: Individual Customer Explanation")
plt.tight_layout()
plt.show()

print("\nPROJECT COMPLETED!")

#  14. SAVE THE MODEL

joblib.dump(final_model, "churn_model.pkl")

print("Model saved successfully!")