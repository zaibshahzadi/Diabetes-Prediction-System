# Diabetes Prediction System

<p align="center">
  <img src="https://github.com/zaibshahzadi/Diabetes-Prediction-System/blob/main/github%20bannar%204.png" width="100%" />
</p>

AI-powered Diabetes Risk Prediction using Machine Learning & Flask

---

## Project Overview

The **Diabetes Prediction System** is a Machine Learning-based web application that predicts whether a patient is likely to have **Diabetes** or **Non-Diabetic** based on key health parameters.

The model is trained using **Random Forest Classifier** and deployed through **Flask** to provide an interactive and user-friendly web interface. Users can enter patient health details and instantly receive prediction results along with model confidence.

This project demonstrates the complete Machine Learning workflow, from data preprocessing and model training to deployment.

---

## Features

-  Predicts diabetes risk (Diabetic / Non-Diabetic)
-  Displays prediction confidence percentage
-  Shows health risk level
-  Generates personalized health suggestions
-  Interactive web interface
-  Fast predictions using trained ML model
-  RESTful API backend
-  Handles imbalanced data using SMOTE
-  Outlier detection and removal

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.12 |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Model Saving | Joblib, Pickle |
| Web Framework | Flask |
| Development Tool | VS Code |
| Notebook | Jupyter Notebook (Colab) |
| Version Control | Git & GitHub |
| Large Files | Git LFS |

---

## Dataset

The model is trained on the **Diabetes Health Indicators Dataset** containing health parameters of patients.

### Dataset Statistics
- **Total Records:** 42,455
- **Features:** 16
- **Target Classes:** 2 (Diabetic / Non-Diabetic)

### Input Features
| Feature | Description |
|---------|-------------|
| **Blood Glucose Level** | Blood glucose concentration (mg/dL) |
| **BMI** | Body Mass Index (kg/m²) |
| **Age** | Age in years |
| **HbA1c Level** | Hemoglobin A1c level (4-9) |
| **Hypertension** | High blood pressure (0/1) |
| **Heart Disease** | History of heart disease (0/1) |
| **Smoking History** | Smoking status |
| **Race** | Race/Ethnicity |
| **Gender** | Gender |
| **Location** | State/Region |
| **Year** | Year of data collection |
| **Skin Thickness** | Triceps skin fold thickness |
| **Insulin** | 2-Hour serum insulin |
| **Pregnancies** | Number of times pregnant |
| **Diabetes Pedigree Function** | Genetic risk factor |

### Target Variable
-  **1** - Diabetic
-  **0** - Non-Diabetic

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Non-Diabetic | 36,413 | 94.98% |
| Diabetic | 1,925 | 5.02% |

---

##  Machine Learning Workflow
Data Collection → Data Preprocessing → Feature Engineering → Handling Missing Values
→ Outlier Detection & Removal → Feature Encoding → Train-Test Split
→ SMOTE Balancing → Feature Scaling → Model Training → Hyperparameter Tuning
→ Model Evaluation → Model Saving → Flask Deployment → Web Interface

---

##  Model Performance

### Best Model: **Random Forest Classifier**

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **91.58%** |
| **Train Accuracy** | **93.51%** |
| **Precision** | 34.33% |
| **Recall** | 74.29% |
| **F1 Score** | 46.96% |
| **AUC-ROC** | **94.06%** |

### Model Performance Comparison

| Machine Learning Model | Accuracy | Recall | F1 Score |
|------------------------|----------|--------|----------|
| **Random Forest** | **97.34%** | **48.05%** | **64.46%** ✅ Best |
| Decision Tree | 95.45% | 57.66% | 55.99% |
| Gradient Boosting | 94.63% | 65.19% | 54.92% |
| K-Nearest Neighbors | 89.76% | 58.70% | 36.54% |
| Logistic Regression | 84.81% | 83.12% | 35.46% |

**Random Forest** achieved the highest accuracy **(91.58%)** and was selected as the final model for deployment.

### Confusion Matrix (Test Set)
Predicted
Non-Diabetic Diabetic
Actual Non-Diabetic 6736 547
Diabetic 99 286

- **True Negatives:** 6,736
- **False Positives:** 547
- **False Negatives:** 99
- **True Positives:** 286

### ROC-AUC Curve
ROC-AUC Score: 0.9406

---

## Feature Importance

Top 10 Most Important Features:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | HbA1c Level | 33.32% |
| 2 | Age | 19.87% |
| 3 | Blood Glucose Level | 19.71% |
| 4 | BMI | 8.16% |
| 5 | Smoking History | 5.51% |
| 6 | Hypertension | 2.26% |
| 7 | Gender (Male) | 1.74% |
| 8 | Heart Disease | 1.71% |
| 9 | Year | 0.97% |
| 10 | Location (Kansas) | 0.48% |

---

### Application Screenshots

Home Page

https://github.com/zaibshahzadi/Diabetes-Prediction-System/blob/main/home%20page%201.png

https://github.com/zaibshahzadi/Diabetes-Prediction-System/blob/main/home%20page%202.png

Diabetic Prediction

https://github.com/zaibshahzadi/Diabetes-Prediction-System/blob/main/dibetic.png

Non-Diabetic Prediction

https://github.com/zaibshahzadi/Diabetes-Prediction-System/blob/main/non-dibetic.png

---

### Future Improvements

Deep Learning implementation (Neural Networks)

Comprehensive health dashboard

Database integration for patient history

User authentication system

Mobile-responsive design

Cloud deployment (AWS/Azure/GCP)

Performance analytics and reporting

Integration with healthcare systems


---

### Author

Zaib Shahzadi

Email: zaibnaveed6@gmail.com

LinkedIn: https://www.linkedin.com/in/zaib-shahzadi-bba376290/
