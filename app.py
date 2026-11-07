import json
import os
import urllib.request

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Diabetes Classification Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 14px;
            border-radius: 12px;
        }
        .app-subtitle {
            color: #777777;
            font-size: 1rem;
            margin-top: -0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

KNN_MODEL_URL = (
    "https://github.com/Akshaya-ProjectsDump/"
    "diabetes-classification-ml/releases/download/v1.0/knn.pkl"
)

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}


@st.cache_resource
def load_project_files():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.path.join(project_dir, "feature_columns.json"),
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    preprocessor = joblib.load(
        os.path.join(project_dir, "preprocessor.pkl")
    )

    model_paths = {
        model_name: os.path.join(project_dir, filename)
        for model_name, filename in MODEL_FILENAMES.items()
    }

    knn_path = model_paths["KNN"]

    if not os.path.exists(knn_path):
        try:
            request = urllib.request.Request(
                KNN_MODEL_URL,
                headers={
                    "User-Agent": "Diabetes-Classification-Streamlit-App"
                },
            )

            with st.spinner(
                "Downloading the KNN model from GitHub Releases..."
            ):
                with urllib.request.urlopen(
                    request,
                    timeout=180,
                ) as response:
                    with open(knn_path, "wb") as model_file:
                        model_file.write(response.read())

        except Exception as download_error:
            if os.path.exists(knn_path):
                os.remove(knn_path)

            raise RuntimeError(
                "Could not download knn.pkl from the GitHub Release. "
                "Check that the release is public and the asset URL is correct. "
                f"Details: {download_error}"
            ) from download_error

    models = {}

    for model_name, model_path in model_paths.items():
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Required model file is missing: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    return models, preprocessor, feature_columns


def prepare_features(data, expected_features):
    missing_features = [
        column for column in expected_features if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded dataset is missing required features: "
            + ", ".join(missing_features)
        )

    features = data[expected_features].copy()

    for column in expected_features:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    if features.isnull().any().any():
        invalid_columns = features.columns[
            features.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing or non-numeric values were found in these required "
            "features: " + ", ".join(invalid_columns)
        )

    return features


def get_model_scores(model, processed_features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_features)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(processed_features)

    return None


def calculate_metrics(actual, predicted, scores=None):
    actual = np.asarray(actual).astype(int)
    predicted = np.asarray(predicted).astype(int)

    auc_value = None

    if scores is not None and len(np.unique(actual)) == 2:
        try:
            auc_value = roc_auc_score(actual, scores)
        except ValueError:
            auc_value = None

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": auc_value,
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


def validate_target(target_series):
    target = pd.to_numeric(target_series, errors="coerce")

    if target.isnull().any():
        raise ValueError(
            "The Diabetes_binary column contains missing or non-numeric values."
        )

    if not set(target.unique()).issubset({0, 1}):
        raise ValueError(
            "The Diabetes_binary column must contain only 0 and 1."
        )

    return target.astype(int)


def make_prediction_results(original_data, predictions):
    results = original_data.copy()
    results["Predicted_Diabetes"] = predictions
    results["Prediction_Label"] = np.where(
        predictions == 1,
        "Diabetic",
        "Non-Diabetic",
    )
    return results


def show_prediction_chart(predictions, title):
    diabetic_count = int(np.sum(np.asarray(predictions) == 1))
    non_diabetic_count = int(np.sum(np.asarray(predictions) == 0))

    chart_data = pd.DataFrame(
        {
            "Prediction": ["Non-Diabetic", "Diabetic"],
            "Number of Records": [non_diabetic_count, diabetic_count],
        }
    ).set_index("Prediction")

    st.subheader(title)

    metric_one, metric_two = st.columns(2)
    metric_one.metric(
        "Predicted Non-Diabetic",
        f"{non_diabetic_count:,}",
    )
    metric_two.metric(
        "Predicted Diabetic",
        f"{diabetic_count:,}",
    )

    st.bar_chart(chart_data, use_container_width=True)


try:
    models, preprocessor, feature_columns = load_project_files()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()

st.title("🩺 Diabetes Classification Using Machine Learning")
st.markdown(
    '<p class="app-subtitle">An interactive machine learning dashboard '
    'using CDC Diabetes Health Indicators.</p>',
    unsafe_allow_html=True,
)

st.info(
    "Academic and educational use only. This application is not a "
    "medical diagnostic tool and must not be used to make medical decisions."
)

with st.sidebar:
    st.header("Dashboard Navigation")
    selected_model_name = st.selectbox(
        "Classification Model",
        list(models.keys()),
    )

    st.divider()
    st.subheader("Project Information")
    st.write("**Dataset:** CDC Diabetes Health Indicators")
    st.write("**Input features:** 21")
    st.write("**Algorithms:** 5")
    st.write("**Problem type:** Binary classification")

selected_model = models[selected_model_name]

dataset_tab, prediction_tab, comparison_tab = st.tabs(
    [
        "📂 Dataset Explorer",
        "🔮 Predictions",
        "📊 Model Comparison",
    ]
)

with dataset_tab:
    st.header("Dataset Explorer")
    st.write(
        "Upload a CSV file containing the 21 health indicator features. "
        "Include Diabetes_binary if you want to evaluate model performance."
    )

    uploaded_file = st.file_uploader(
        "Upload diabetes health indicators CSV",
        type=["csv"],
        key="dataset_uploader",
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state["uploaded_dataset"] = data

            total_rows, total_columns = data.shape
            total_missing = int(data.isnull().sum().sum())
            total_duplicates = int(data.duplicated().sum())

            st.success("CSV file uploaded successfully.")

            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Total Records", f"{total_rows:,}"
            )
            metric_columns[1].metric(
                "Total Columns", total_columns
            )
            metric_columns[2].metric(
                "Missing Values", f"{total_missing:,}"
            )
            metric_columns[3].metric(
                "Duplicate Records", f"{total_duplicates:,}"
            )

            preview_tab, missing_tab, columns_tab = st.tabs(
                [
                    "Dataset Preview",
                    "Missing Values",
                    "Column Information",
                ]
            )

            with preview_tab:
                st.subheader("Dataset Preview")
                st.dataframe(
                    data.head(20),
                    use_container_width=True,
                )
                st.caption(
                    f"Showing the first {min(20, len(data))} records."
                )

            with missing_tab:
                st.subheader("Missing Value Analysis")

                missing_summary = pd.DataFrame(
                    {
                        "Column": data.columns,
                        "Missing Values": data.isnull().sum().values,
                        "Missing Percentage": (
                            data.isnull().mean().values * 100
                        ).round(2),
                    }
                )

                missing_summary = missing_summary[
                    missing_summary["Missing Values"] > 0
                ]

                if missing_summary.empty:
                    st.success("No missing values were found.")
                else:
                    st.dataframe(
                        missing_summary,
                        use_container_width=True,
                    )

            with columns_tab:
                st.subheader("Column Information")

                column_summary = pd.DataFrame(
                    {
                        "Column Name": data.columns,
                        "Data Type": data.dtypes.astype(str).values,
                        "Unique Values": data.nunique().values,
                    }
                )

                st.dataframe(
                    column_summary,
                    use_container_width=True,
                )

            if "Diabetes_binary" in data.columns:
                st.subheader("Actual Target Distribution")

                try:
                    target_values = validate_target(
                        data["Diabetes_binary"]
                    )

                    target_counts = (
                        target_values.map(
                            {0: "Non-Diabetic", 1: "Diabetic"}
                        )
                        .value_counts()
                        .rename_axis("Class")
                        .to_frame("Number of Records")
                    )

                    st.bar_chart(
                        target_counts,
                        use_container_width=True,
                    )

                except ValueError as target_error:
                    st.warning(str(target_error))

        except Exception as error:
            st.error(f"Unable to read the uploaded CSV: {error}")
    else:
        st.session_state.pop("uploaded_dataset", None)
        st.info("Upload a CSV file to explore the dataset.")

uploaded_data = st.session_state.get("uploaded_dataset")

prepared_features = None
processed_features = None
actual_target = None
dataset_error = None

if uploaded_data is not None:
    try:
        prepared_features = prepare_features(
            uploaded_data,
            feature_columns,
        )
        processed_features = preprocessor.transform(
            prepared_features
        )

        if "Diabetes_binary" in uploaded_data.columns:
            actual_target = validate_target(
                uploaded_data["Diabetes_binary"]
            )

    except Exception as error:
        dataset_error = str(error)

with prediction_tab:
    st.header("Generate Predictions")
    st.write(f"Selected model: **{selected_model_name}**")

    if uploaded_data is None:
        st.info(
            "Upload a CSV file in the Dataset Explorer tab to generate predictions."
        )
    elif dataset_error:
        st.error(dataset_error)
    else:
        st.write(
            f"Ready to generate predictions for "
            f"**{len(uploaded_data):,} records**."
        )

        if st.button(
            "Generate Predictions",
            type="primary",
            key="generate_selected_predictions",
        ):
            try:
                predictions = selected_model.predict(
                    processed_features
                )

                prediction_results = make_prediction_results(
                    uploaded_data,
                    predictions,
                )

                st.session_state["selected_prediction_results"] = (
                    prediction_results
                )
                st.session_state["selected_predictions"] = predictions
                st.session_state["selected_prediction_model"] = (
                    selected_model_name
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

        saved_predictions = st.session_state.get(
            "selected_predictions"
        )

        if saved_predictions is not None:
            saved_model = st.session_state.get(
                "selected_prediction_model"
            )

            if saved_model == selected_model_name:
                prediction_results = st.session_state[
                    "selected_prediction_results"
                ]

                st.subheader("Prediction Results")
                st.dataframe(
                    prediction_results.head(50),
                    use_container_width=True,
                )

                show_prediction_chart(
                    saved_predictions,
                    "Predicted Diabetes Class Distribution",
                )

                csv_output = prediction_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Report (CSV)",
                    data=csv_output,
                    file_name="diabetes_prediction_report.csv",
                    mime="text/csv",
                    key="download_selected_predictions",
                )

                if actual_target is not None:
                    st.subheader("Model Evaluation")

                    selected_scores = get_model_scores(
                        selected_model,
                        processed_features,
                    )

                    selected_metrics = calculate_metrics(
                        actual_target,
                        saved_predictions,
                        selected_scores,
                    )

                    first_metrics = st.columns(3)

                    for column, metric_name in zip(
                        first_metrics,
                        ["Accuracy", "AUC", "Precision"],
                    ):
                        metric_value = selected_metrics[metric_name]
                        column.metric(
                            metric_name,
                            (
                                f"{metric_value:.4f}"
                                if metric_value is not None
                                else "N/A"
                            ),
                        )

                    second_metrics = st.columns(3)

                    for column, metric_name in zip(
                        second_metrics,
                        ["Recall", "F1", "MCC"],
                    ):
                        metric_value = selected_metrics[metric_name]
                        column.metric(
                            "F1 Score" if metric_name == "F1" else metric_name,
                            f"{metric_value:.4f}",
                        )

                    matrix = confusion_matrix(
                        actual_target,
                        saved_predictions,
                        labels=[0, 1],
                    )

                    figure, axis = plt.subplots(figsize=(6, 4))
                    display = ConfusionMatrixDisplay(
                        confusion_matrix=matrix,
                        display_labels=[
                            "Non-Diabetic",
                            "Diabetic",
                        ],
                    )
                    display.plot(
                        cmap="Blues",
                        ax=axis,
                        colorbar=False,
                    )
                    axis.set_title(
                        f"Confusion Matrix - {selected_model_name}"
                    )
                    st.pyplot(figure)
                    plt.close(figure)

                    report = classification_report(
                        actual_target,
                        saved_predictions,
                        target_names=[
                            "Non-Diabetic",
                            "Diabetic",
                        ],
                        output_dict=True,
                        zero_division=0,
                    )

                    st.subheader("Classification Report")
                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4),
                        use_container_width=True,
                    )

            else:
                st.info(
                    "Select Generate Predictions to run the newly selected model."
                )

with comparison_tab:
    st.header("Interactive Model Comparison")
    st.write(
        "Compare all five trained classifiers using the six evaluation metrics."
    )

    if uploaded_data is None:
        st.info(
            "Upload a CSV file in the Dataset Explorer tab first."
        )
    elif dataset_error:
        st.error(dataset_error)
    elif actual_target is None:
        st.warning(
            "The uploaded CSV does not contain Diabetes_binary. "
            "Model evaluation and ROC comparison require actual target labels."
        )
    else:
        st.write(
            "Every model is evaluated against the actual Diabetes_binary "
            "labels in the uploaded dataset."
        )

        if st.button(
            "Compare All Five Models",
            type="primary",
            key="compare_all_models",
        ):
            comparison_rows = []
            comparison_predictions = {}
            comparison_scores = {}

            progress = st.progress(0)
            status = st.empty()

            try:
                model_items = list(models.items())

                for index, (model_name, model) in enumerate(model_items):
                    status.write(f"Evaluating {model_name}...")

                    predictions = model.predict(processed_features)
                    scores = get_model_scores(
                        model,
                        processed_features,
                    )

                    metrics = calculate_metrics(
                        actual_target,
                        predictions,
                        scores,
                    )

                    comparison_rows.append(
                        {"Model": model_name, **metrics}
                    )
                    comparison_predictions[model_name] = predictions
                    comparison_scores[model_name] = scores

                    progress.progress(
                        (index + 1) / len(model_items)
                    )

                comparison_table = pd.DataFrame(comparison_rows)

                st.session_state["comparison_table"] = comparison_table
                st.session_state["comparison_predictions"] = (
                    comparison_predictions
                )
                st.session_state["comparison_scores"] = comparison_scores
                st.session_state["comparison_dataset_rows"] = len(
                    uploaded_data
                )

                status.success(
                    "All five models evaluated successfully."
                )

            except Exception as error:
                status.empty()
                st.error(f"Model comparison failed: {error}")

        comparison_table = st.session_state.get("comparison_table")

        if (
            comparison_table is not None
            and st.session_state.get("comparison_dataset_rows")
            == len(uploaded_data)
        ):
            st.subheader("Performance Comparison Table")

            st.dataframe(
                comparison_table.round(4),
                use_container_width=True,
                hide_index=True,
            )

            metric_options = [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "MCC",
            ]

            selected_metric = st.selectbox(
                "Choose a metric to compare:",
                metric_options,
                index=4,
                key="comparison_metric",
            )

            chart_data = (
                comparison_table[["Model", selected_metric]]
                .dropna()
                .set_index("Model")
            )

            st.subheader(
                f"Model Comparison by {selected_metric}"
            )
            st.bar_chart(
                chart_data,
                use_container_width=True,
            )

            comparison_csv = comparison_table.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Model Comparison Report",
                data=comparison_csv,
                file_name="diabetes_model_comparison.csv",
                mime="text/csv",
                key="download_model_comparison",
            )

            st.subheader("ROC Curve Comparison")

            scores_by_model = st.session_state.get(
                "comparison_scores",
                {},
            )

            if len(np.unique(actual_target)) == 2:
                figure, axis = plt.subplots(figsize=(8, 6))
                plotted_any = False

                for model_name, scores in scores_by_model.items():
                    if scores is None:
                        continue

                    false_positive_rate, true_positive_rate, _ = (
                        roc_curve(actual_target, scores)
                    )
                    model_auc = auc(
                        false_positive_rate,
                        true_positive_rate,
                    )

                    axis.plot(
                        false_positive_rate,
                        true_positive_rate,
                        label=f"{model_name} (AUC = {model_auc:.3f})",
                    )
                    plotted_any = True

                axis.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    label="No-skill baseline",
                )
                axis.set_xlabel("False Positive Rate")
                axis.set_ylabel("True Positive Rate")
                axis.set_title("ROC Curve Comparison")
                axis.legend(loc="lower right")
                axis.grid(alpha=0.25)

                if plotted_any:
                    st.pyplot(figure)
                else:
                    st.warning(
                        "None of the loaded models provides probability "
                        "or decision scores required to draw ROC curves."
                    )

                plt.close(figure)
            else:
                st.warning(
                    "ROC curves require both target classes in the uploaded dataset."
                )

            st.subheader("Model-wise Prediction Distribution")

            all_predictions = st.session_state.get(
                "comparison_predictions",
                {},
            )

            distribution_rows = []

            for model_name, predictions in all_predictions.items():
                distribution_rows.append(
                    {
                        "Model": model_name,
                        "Non-Diabetic": int(
                            np.sum(np.asarray(predictions) == 0)
                        ),
                        "Diabetic": int(
                            np.sum(np.asarray(predictions) == 1)
                        ),
                    }
                )

            if distribution_rows:
                distribution_table = pd.DataFrame(
                    distribution_rows
                ).set_index("Model")

                st.bar_chart(
                    distribution_table,
                    use_container_width=True,
                )

st.divider()
st.caption(
    "Diabetes Classification Using Machine Learning | "
    "CDC Diabetes Health Indicators | Academic project"
)
