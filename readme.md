# 📊 HR Analytics: Employee Attrition Analysis & Prediction

An end-to-end **HR Analytics and Machine Learning project** that analyzes why employees leave an organization and predicts which active employees may be at higher risk of attrition.

The project combines **MySQL, Python, Machine Learning, and Power BI** to build a complete analytics pipeline — from data storage and exploratory analysis to predictive modeling and interactive business dashboards.

---

## 📌 Project Overview

Employee attrition is an important HR challenge because unexpected employee turnover can increase recruitment costs, reduce productivity, and affect team performance.

This project addresses two key questions:

1. **Why are employees leaving?**
2. **Which currently active employees may be at higher risk of leaving?**

The project includes:

- SQL-based data analysis using **MySQL**
- Exploratory Data Analysis using **Python**
- Statistical hypothesis testing
- Machine Learning using **Logistic Regression and Random Forest**
- Hyperparameter tuning with **GridSearchCV**
- Threshold optimization for attrition prediction
- Attrition probability scoring for active employees
- Interactive **Power BI dashboards**

---

## 📂 Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Source:** Kaggle

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- **Employees:** 1,470
- **Features:** 35
- **Target:** `Attrition`
- `Yes` = Employee left
- `No` = Employee stayed

> **Note:** This is a sample/synthetic dataset and does not represent a specific real-world organization.

---

# 📊 Dashboard Preview

## Page 1 — Attrition Overview

![Attrition Overview](https://github.com/sahibsingh25012005-ui/hr_attrition_prediction_analysis/blob/main/Images/Hr_Attrition_Overview.png)

This page focuses on **historical employee attrition**, including:

- Overall attrition rate
- Employee demographics
- Department and job-role attrition
- Income analysis
- Job satisfaction
- Employee tenure
- Other HR-related patterns

---

## Page 2 — Predicted Attrition Risk

![Predicted Attrition Risk](images/predicted_attrition.png)

This page focuses on the **active workforce** and the employees identified by the ML model as having a higher predicted probability of attrition.

It includes:

- Predicted attrition count
- Attrition probability
- Department-wise risk
- Job-role risk
- Overtime analysis
- Tenure-based risk patterns

---

# 🔄 Project Workflow

```text
                  MySQL
                    │
                    ▼
             hr_attrition
                    │
                    ▼
        Python: EDA + KPIs
                    │
                    ▼
         Hypothesis Testing
                    │
                    ▼
       Feature Engineering
                    │
                    ▼
       Machine Learning Models
                    │
            ┌───────┴───────┐
            ▼               ▼
     Random Forest    Logistic Regression
            │               │
            └───────┬───────┘
                    ▼
             Model Evaluation
                    │
                    ▼
          Threshold Optimization
                    │
                    ▼
       Active Employee Prediction
                    │
                    ▼
             MySQL Database
                    │
                    ▼
          predicted_dataset
                    │
                    ▼
              Power BI
```

---

# 🛠️ Project Workflow

### 1. Data Storage

The raw HR dataset is stored in MySQL in the:

```text
hr_attrition
```

table.

SQL was used to perform analytical queries and calculate HR KPIs.

---

### 2. Exploratory Data Analysis

Python was used to explore employee characteristics and identify potential attrition patterns.

Libraries used:

- Pandas
- NumPy
- Matplotlib
- Seaborn

The analysis included:

- Attrition distribution
- Income analysis
- Job satisfaction
- Department analysis
- Job-role analysis
- Tenure analysis
- Overtime analysis

---

### 3. Hypothesis Testing

A **Welch's independent two-sample t-test** was performed to investigate whether average monthly income differed between employees who stayed and employees who left.

Results:

| Group | Average Monthly Income |
|---|---:|
| Stayed | $6,833 |
| Left | $4,787 |

**Welch's t-test:**

```text
t = 7.48
p ≈ 4.4e-13
```

The result provides statistical evidence of a difference in average monthly income between the two groups.

> Statistical significance does not establish that income itself causes attrition.

---

# 📈 Key Historical Findings

The historical analysis uses all **1,470 employees**.

| Metric | Value |
|---|---:|
| Total Employees | 1,470 |
| Active Employees | 1,233 |
| Employees Who Left | 237 |
| Overall Attrition Rate | **16.1%** |
| Average Tenure | 7.01 years |
| Average Tenure in Current Role | 4.23 years |
| Average Monthly Income | $6,503 |

### Job Role

Sales Representatives have the highest observed attrition rate among job roles:

- **Sales Representatives:** 39.76%
- **Laboratory Technicians:** 23.94%
- **Human Resources:** 23.08%

### Department

Department-level attrition rates:

- **Sales:** 20.63%
- **Human Resources:** 19.05%
- **Research & Development:** 13.84%

### Monthly Income

Employees earning below $3,000 per month had an observed attrition rate of:

**28.61%**

compared with:

**8.90%**

for employees earning above $10,000 per month.

### Job Satisfaction

Employees in the lowest job-satisfaction group had an observed attrition rate of:

**22.84%**

compared with:

**11.33%**

for employees in the highest satisfaction group.

---

# 🤖 Machine Learning

## Objective

The machine learning model predicts the probability that an employee will leave the organization.

### Target Variable

```text
Attrition
```

Encoding:

```text
Yes → 1
No  → 0
```

---

## 🧹 Data Preprocessing

The following non-informative columns were removed:

```text
EmployeeNumber
EmployeeCount
StandardHours
Over18
```

`Gender` was excluded from the predictive model to avoid using a protected demographic attribute as a predictor.

### Encoding

- `OverTime` → binary encoding
- `Education` → ordinal encoding
- Remaining categorical variables → one-hot encoding
- `drop_first=True` used for one-hot encoding

### Feature Scaling

Numeric features were standardized using:

```python
StandardScaler()
```

The scaler was fitted only on the training data to avoid data leakage.

### Train/Test Split

```text
Training set: 80%
Testing set: 20%
random_state = 42
```

---

# 🧠 Models Compared

Two classification models were evaluated.

| Model | Tuning | Result |
|---|---|---|
| Random Forest | GridSearchCV, 5-fold CV, scoring = recall | Accuracy: 0.87, Recall for leavers: 0.10 |
| Logistic Regression | GridSearchCV, 5-fold CV, scoring = F1 | Selected as final model |

### Logistic Regression Best Parameters

```python
C = 1
penalty = "l1"
solver = "liblinear"
```

---

# 🎯 Threshold Optimization

The dataset is imbalanced, with only about **16% of employees being leavers**.

Using the default classification threshold of `0.50` can result in too many actual leavers being classified as staying.

Therefore, different probability thresholds were evaluated.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.50 | 0.56 | 0.36 | 0.44 |
| 0.45 | 0.50 | 0.41 | 0.45 |
| **0.40** | **0.50** | **0.49** | **0.49** |

The final model uses a:

```text
Decision Threshold = 0.40
```

This threshold was selected based on the observed trade-off between precision and recall.

---

# 📊 Confusion Matrix

Test-set performance at the `0.40` threshold:

**Test set size: 294 employees**

| | Predicted Stay | Predicted Leave |
|---|---:|---:|
| **Actual Stay** | 227 | 28 |
| **Actual Leave** | 20 | 19 |

From this result:

- True Negatives = 227
- False Positives = 28
- False Negatives = 20
- True Positives = 19

The model identified **19 of the 39 actual leavers** in the test set.

---

# 🔮 Predicting Attrition for Active Employees

After evaluating the model, it was applied only to employees whose historical attrition status was:

```text
Attrition = No
```

This represents the **1,233 currently active employees** in the dataset.

For each active employee, the model generates:

```text
Attrition_probability
Predict_attrition
```

The probability is stored as a percentage and the prediction is based on the selected `0.40` threshold.

The results are then written back to MySQL in:

```text
pred
