
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Classification Using Machine Learning")
st.write(
    "Upload a diabetes health indicators CSV file, "
    "select a classification model, and generate predictions."
)

st.info(
    "This application is for educational purposes only. "
    "It is not a medical diagnostic tool."
)

# --------------------------------------------------
# 2. LOAD SAVED MODELS AND PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_project_files():

    with open("feature_columns.json", "r") as file:
        feature_columns = json.load(file)

    preprocessor = joblib.load("preprocessor.pkl")

    model_files = {
        "Logistic Regression": "saved_models/logistic_regression.pkl",
        "Decision Tree": "saved_models/decision_tree.pkl",
        "KNN": "saved_models/knn.pkl",
        "Naive Bayes": "saved_models/naive_bayes.pkl",
        "SVM": "saved_models/svm.pkl"
    }

    models = {}

    for model_name, file_path in model_files.items():
        models[model_name] = joblib.load(file_path)

    return models, preprocessor, feature_columns


try:
    models, preprocessor, feature_columns = load_project_files()
    st.success("All five classification models loaded successfully!")

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# 3. MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a machine learning algorithm:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.write("Selected model:", selected_model_name)


# --------------------------------------------------
# 4. CSV FILE UPLOAD
# --------------------------------------------------

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your diabetes CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        st.write("Dataset preview:")
        st.dataframe(data.head())

        st.write("Dataset shape:", data.shape)

        # --------------------------------------------------
        # 5. VALIDATE INPUT FEATURES
        # --------------------------------------------------

        has_target = "Diabetes_binary" in data.columns

        missing_features = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing_features:

            st.error(
                "The uploaded CSV is missing these required features: "
                + ", ".join(missing_features)
            )

            st.stop()

        # Select features in the exact saved training order
        X = data[feature_columns].copy()

        # Convert input columns to numeric values
        for column in feature_columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        if X.isnull().any().any():

            st.error(
                "The input data contains missing or non-numeric "
                "values in the required features. Please correct "
                "the CSV and upload it again."
            )

            st.stop()

        # --------------------------------------------------
        # 6. PREPROCESS INPUT DATA
        # --------------------------------------------------

        X_processed = preprocessor.transform(X)

        st.success("Input data preprocessing completed!")

        # --------------------------------------------------
        # 7. GENERATE PREDICTIONS
        # --------------------------------------------------

        if st.button("Generate Predictions"):

            predictions = selected_model.predict(X_processed)

            results = data.copy()

            results["Predicted_Diabetes"] = predictions

            results["Prediction_Label"] = np.where(
                predictions == 1,
                "Diabetic",
                "Non-Diabetic"
            )

            st.subheader("Prediction Results")

            st.dataframe(results.head(20))

            diabetic_count = int(np.sum(predictions == 1))
            non_diabetic_count = int(np.sum(predictions == 0))

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Diabetic",
                diabetic_count
            )

            col2.metric(
                "Predicted Non-Diabetic",
                non_diabetic_count
            )

            # --------------------------------------------------
            # 8. DOWNLOAD PREDICTION RESULTS
            # --------------------------------------------------

            csv_output = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="diabetes_predictions.csv",
                mime="text/csv"
            )

            # --------------------------------------------------
            # 9. EVALUATION IF ACTUAL TARGET IS AVAILABLE
            # --------------------------------------------------

            if has_target:

                st.subheader("Model Evaluation")

                y_true = pd.to_numeric(
                    data["Diabetes_binary"],
                    errors="coerce"
                )

                if y_true.isnull().any() or not set(y_true.unique()).issubset({0, 1}):

                    st.warning(
                        "The actual target column must contain only "
                        "valid binary values: 0 and 1."
                    )

                else:

                    y_true = y_true.astype(int)

                    # Obtain scores for AUC
                    if hasattr(selected_model, "predict_proba"):

                        y_scores = selected_model.predict_proba(
                            X_processed
                        )[:, 1]

                    elif hasattr(selected_model, "decision_function"):

                        y_scores = selected_model.decision_function(
                            X_processed
                        )

                    else:
                        y_scores = None

                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(
                        y_true, predictions, zero_division=0
                    )
                    recall = recall_score(
                        y_true, predictions, zero_division=0
                    )
                    f1 = f1_score(
                        y_true, predictions, zero_division=0
                    )
                    mcc = matthews_corrcoef(y_true, predictions)

                    if y_scores is not None and len(np.unique(y_true)) == 2:
                        auc = roc_auc_score(y_true, y_scores)
                    else:
                        auc = None

                    # Display evaluation metrics
                    c1, c2, c3 = st.columns(3)

                    c1.metric("Accuracy", f"{accuracy:.4f}")
                    c2.metric(
                        "AUC",
                        f"{auc:.4f}" if auc is not None else "N/A"
                    )
                    c3.metric("Precision", f"{precision:.4f}")

                    c4, c5, c6 = st.columns(3)

                    c4.metric("Recall", f"{recall:.4f}")
                    c5.metric("F1 Score", f"{f1:.4f}")
                    c6.metric("MCC", f"{mcc:.4f}")

                    # --------------------------------------------------
                    # 10. CONFUSION MATRIX
                    # --------------------------------------------------

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_true,
                        predictions,
                        labels=[0, 1]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    display = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=["Non-Diabetic", "Diabetic"]
                    )

                    display.plot(
                        cmap="Blues",
                        ax=ax,
                        colorbar=False
                    )

                    ax.set_title(
                        f"Confusion Matrix - {selected_model_name}"
                    )

                    st.pyplot(fig)
                    plt.close(fig)

                    # --------------------------------------------------
                    # 11. CLASSIFICATION REPORT
                    # --------------------------------------------------

                    st.subheader("Classification Report")

                    report = classification_report(
                        y_true,
                        predictions,
                        target_names=["Non-Diabetic", "Diabetic"],
                        output_dict=True,
                        zero_division=0
                    )

                    report_df = pd.DataFrame(report).transpose()

                    st.dataframe(report_df.round(4))

            else:

                st.info(
                    "No Diabetes_binary target column was found. "
                    "Predictions are available, but evaluation metrics "
                    "and the confusion matrix cannot be calculated."
                )

    except Exception as error:

        st.error(f"An error occurred while processing the CSV: {error}")
