
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


# PERFORMING THE PAGE CONFIGURATION

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


# LOADING THE SAVED MODELS AND THE PREPROCESSOR


# GitHub Release asset used when knn.pkl is not present in the project folder.
KNN_MODEL_URL = "https://github.com/Akshaya-ProjectsDump/diabetes-classification-ml/releases/download/v1.0/knn.pkl"


@st.cache_resource
def load_project_files():

    # Resolve paths relative to this app.py file, not the terminal's current folder.
    project_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(project_dir, "feature_columns.json"), "r") as file:
        feature_columns = json.load(file)

    preprocessor = joblib.load(os.path.join(project_dir, "preprocessor.pkl"))

    model_files = {
    "Logistic Regression": os.path.join(project_dir, "logistic_regression.pkl"),
    "Decision Tree": os.path.join(project_dir, "decision_tree.pkl"),
    "KNN": os.path.join(project_dir, "knn.pkl"),
    "Naive Bayes": os.path.join(project_dir, "naive_bayes.pkl"),
    "SVM": os.path.join(project_dir, "svm.pkl")
}

    # Downloading the KNN model from the public GitHub Release if it is not stored locally.
    knn_path = model_files["KNN"]
    if not os.path.exists(knn_path):
        try:
            import urllib.request

            request = urllib.request.Request(
                KNN_MODEL_URL,
                headers={"User-Agent": "Diabetes-Classification-Streamlit-App"}
            )
            with st.spinner("Downloading the KNN model from GitHub Release..."):
                with urllib.request.urlopen(request, timeout=180) as response:
                    with open(knn_path, "wb") as model_file:
                        model_file.write(response.read())
        except Exception as download_error:
            if os.path.exists(knn_path):
                os.remove(knn_path)
            raise RuntimeError(
                "Could not download knn.pkl from the GitHub Release. "
                "Check that the release is public and the asset link is correct. "
                f"Details: {download_error}"
            ) from download_error

    models = {}

    for model_name, file_path in model_files.items():
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Required model file is missing: {file_path}"
            )
        models[model_name] = joblib.load(file_path)

    return models, preprocessor, feature_columns


try:
    models, preprocessor, feature_columns = load_project_files()
    st.success("All five classification models loaded successfully!")

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# PERFROMING THE MODEL SELECTION

st.subheader("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a machine learning algorithm:",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.write("Selected model:", selected_model_name)



# PERFORMING THE CSV FILE UPLOAD

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your diabetes CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

# DATASET OVERVIEW

st.subheader("Dataset Overview")

# Display important dataset statistics
total_rows, total_columns = data.shape
total_missing = int(data.isnull().sum().sum())
total_duplicates = int(data.duplicated().sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_rows:,}")
col2.metric("Total Features / Columns", total_columns)
col3.metric("Missing Values", total_missing)
col4.metric("Duplicate Records", total_duplicates)

# Create interactive tabs
tab1, tab2, tab3 = st.tabs([
    "Dataset Preview",
    "Missing Values",
    "Column Information"
])

# TAB 1: Dataset preview
with tab1:

    st.write("First 10 records of the uploaded dataset:")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

# TAB 2: Missing value analysis
with tab2:

    missing_summary = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": data.isnull().sum().values,
        "Missing Percentage": (
            data.isnull().mean().values * 100
        ).round(2)
    })

    missing_summary = missing_summary[
        missing_summary["Missing Values"] > 0
    ]

    if missing_summary.empty:

        st.success("No missing values found in the dataset.")

    else:

        st.dataframe(
            missing_summary,
            use_container_width=True
        )

# TAB 3: Column information
with tab3:

    column_summary = pd.DataFrame({
        "Column Name": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Unique Values": data.nunique().values
    })

    st.dataframe(
        column_summary,
        use_container_width=True
    )


        
        # VALIDATING INPUT FEATURES
      

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

        # Converting input columns to numeric values
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

      
        # PREPROCESSING THE INPUT DATA
        
        X_processed = preprocessor.transform(X)

        st.success("Input data preprocessing completed!")

        
        #GENERATING THE PREDICTIONS
        

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

            
            # DOWNLOADING THE PREDICTION RESULTS
            

            csv_output = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="diabetes_predictions.csv",
                mime="text/csv"
            )

         
            # EVALUATING IF ACTUAL TARGET IS AVAILABLE
           

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

                    
                    # CONSTRUCTING THE CONFUSION MATRIX
                    

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

                    
                    # CONSTRUCTING THE CLASSIFICATION REPORT
                   

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
