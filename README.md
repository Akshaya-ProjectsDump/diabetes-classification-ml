# Diabetes Classification Using Machine Learning

## 1. Problem Statement

This project develops a machine learning-based diabetes classification system using health indicators from the public CDC Diabetes Health Indicators dataset.

The objective is to predict whether an individual is diabetic or non-diabetic based on health-related indicators.

The project implements five classification algorithms and compares their performance using multiple evaluation metrics to identify the most suitable model for the selected dataset.

## 2. Dataset Description

- **Dataset:** CDC Diabetes Health Indicators
- **Source:** UCI Machine Learning Repository
- **Number of records:** 253,680
- **Number of input features:** 21
- **Target variable:** `Diabetes_binary`
- **Problem type:** Binary classification
- **Target classes:** 0 – Non-Diabetic, 1 – Diabetic

The dataset contains health-related indicators such as BMI, general health, physical activity, age, smoking status and other lifestyle-related factors.

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/Akshaya-ProjectsDump/diabetes-classification-ml

The repository contains the machine learning source code, trained models, preprocessing files, test dataset, Streamlit application and project documentation.

The KNN model is provided as a downloadable asset in the repository's GitHub Releases.

## 4. Machine Learning Models Used

Five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

## 5. Model Evaluation and Comparison
### Training and Preprocessing Methodology
The dataset was prepared for binary classification using the Diabetes_binary target variable. 
•	The input features were separated from the target variable before model training.
•	The preprocessing pipeline was used to prepare the input features for the classification algorithms. 
•	The dataset was then divided into training and testing subsets at a split ratio of 80% and 20% i.e. 50,736 records and 22 columns, including 21 input features and the target variable out of the original dataset of 253,680 records.
•	Five classification algorithms were trained and evaluated using the same evaluation metrics to compare their performance.
The trained models and preprocessing components were saved for reuse in the Streamlit application.


The models are evaluated using six performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7316 | 0.8196 | 0.3108 | 0.7608 | 0.4413 | 0.3563 |
| Decision Tree | 0.7040 | 0.8063 | 0.2917 | 0.7877 | 0.4258 | 0.3417 |
| KNN | 0.8490 | 0.7266 | 0.4193 | 0.2176 | 0.2865 | 0.2257 |
| Naive Bayes | 0.7722 | 0.7803 | 0.3198 | 0.5639 | 0.4082 | 0.2974 |
| Support Vector Machine | 0.7275 | 0.8194 | 0.3083 | 0.7686 | 0.4400 | 0.3559 |

### Observations on Model Performance

**Logistic Regression**

Achieved the highest F1-score of 0.4413 and AUC of 0.8196. It provides a good balance between precision and recall among the five models.

**Decision Tree**

Achieved the highest recall of 0.7877. It identifies a larger proportion of actual diabetic cases compared with the other models.

**K-Nearest Neighbors (KNN)**

Achieved the highest accuracy of 0.8490 and precision of 0.4193. However, its recall of 0.2176 is low, indicating that it misses many actual diabetic cases.

**Naive Bayes**

Achieved an accuracy of 0.7722 and recall of 0.5639. It provides moderate classification performance compared with the other models.

**Support Vector Machine (SVM)**

Achieved an AUC of 0.8194 and F1-score of 0.4400. Its performance is very close to Logistic Regression.

### Overall Winner

Logistic Regression is selected as the overall winner based on its highest F1-score and AUC.

However, Decision Tree achieved the highest recall, which is an important consideration when identifying diabetic cases. Therefore, model selection depends on the evaluation metric and project objective.

## 6. Streamlit Application

**Live Streamlit Application:**

https://cdc-ml-diabetes-classifier.streamlit.app

The trained diabetes classification models have been deployed using Streamlit Community Cloud. The application provides an interactive interface for exploring the dataset, generating predictions and comparing the performance of five machine learning classification algorithms.

### Features of the Application

**Dataset Explorer**
- Upload a CSV file containing diabetes health indicators.
- Preview uploaded dataset records.
- View the total number of records and columns.
- Analyse missing values and duplicate records.
- View column names, data types and unique values.
- Visualise the actual target class distribution when target labels are available.

**Diabetes Prediction**
- Select one of the five trained classification models.
- Generate predictions using the selected model.
- View predicted diabetic and non-diabetic counts.
- Visualise prediction distributions.
- View evaluation metrics when actual target labels are available.
- View the confusion matrix and classification report.
- Download prediction results as a CSV file.

**Interactive Model Comparison**
- Evaluate all five trained classification models on the uploaded labelled dataset.
- Compare Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).
- Select an evaluation metric for visual comparison.
- View model-wise prediction distributions.
- Download the model comparison report.

**ROC Curve Comparison**
- Visualise ROC curves for the five classification models.
- Compare their AUC performance.
- View the no-skill baseline for reference.

The application is intended for academic and educational purposes only and is not a medical diagnostic tool.

## 7. Project Structure

```text
diabetes-classification-ml/
│
├── ML Assignment -2.ipynb
├── app.py
├── README.md
├── requirements.txt
├── preprocessor.pkl
├── feature_columns.json
├── test_data.csv
├── logistic_regression.pkl
├── decision_tree.pkl
├── naive_bayes.pkl
├── svm.pkl
│
└── KNN model:
    └── Available in GitHub Releases
```

## 8. Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 9. Run the Application

Run the following command:

```bash
streamlit run app.py
```

## 10. Disclaimer

This project is intended for academic and educational purposes only. It is not a medical diagnostic tool and should not be used to make medical decisions.
