# Diabetes Classification Using Machine Learning

## 1. Project Overview
This project develops a machine learning-based diabetes classification system using health indicators from the CDC Diabetes Health Indicators dataset.

The objective is to predict whether an individual is diabetic or non-diabetic based on health-related indicators.

## 2. Dataset Information
- Dataset: CDC Diabetes Health Indicators
- Source: UCI Machine Learning Repository
- Number of records: 253,680
- Number of input features: 21
- Target variable: `Diabetes_binary`
- Problem type: Binary classification

## 3. Machine Learning Algorithms
Five classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics
The models are evaluated using:
- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion matrices, ROC curves, classification reports, and execution-time comparisons are also included.

## 5. Streamlit Application
The application provides the following features:
- Upload a CSV dataset.
- Select one of the five trained classification models.
- Generate diabetes predictions.
- View prediction counts.
- View model evaluation metrics when actual target labels are available.
- View the confusion matrix and classification report when actual target labels are available.
- Download prediction results.

## 6. Project Structure

```text
diabetes-classification-ml/
├── app.py
├── requirements.txt
├── README.md
├── preprocessor.pkl
├── feature_columns.json
├── test_data.csv
└── saved_models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## 7. Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

```bash
streamlit run app.py
```

## 9. Disclaimer
This project is intended for academic and educational purposes only. It is not a medical diagnostic tool.
