\# HR Analytics: Employee Attrition Analysis \& Prediction



An end-to-end analytics project that explores \*\*why employees leave\*\* and \*\*who is likely to leave next\*\*. It combines SQL (MySQL), Python (pandas, scikit-learn) and a two-page interactive \*\*Power BI\*\* dashboard.



> \*\*Data source:\*\* \[IBM HR Analytics Employee Attrition \& Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) sample dataset (1,470 employees, 35 features).



\---



\## Dashboard Preview



\### Page 1: Attrition Overview

!\[Attrition Overview](images/attrition\_overview.png)



\### Page 2: Predicted Attrition Risk

!\[Predicted Attrition Risk](images/predicted\_attrition.png)



\---



\## Project Workflow



```

&#x20;MySQL (hr\_attrition)

&#x20;       │

&#x20;       ▼

&#x20;Python: EDA, KPIs, hypothesis test

&#x20;       │

&#x20;       ▼

&#x20;Feature engineering → Logistic Regression (threshold-tuned)

&#x20;       │

&#x20;       ▼

&#x20;Attrition probabilities for active employees

&#x20;       │

&#x20;       ▼

&#x20;MySQL (predicted\_dataset)  →  Power BI dashboard

```



1\. \*\*Data storage:\*\* the raw dataset lives in a MySQL table `hr\_attrition`.

2\. \*\*Analysis:\*\* SQL queries and pandas compute KPIs; seaborn/matplotlib explore distributions.

3\. \*\*Hypothesis testing:\*\* a Welch's t-test checks whether income differs between leavers and stayers.

4\. \*\*Modelling:\*\* Random Forest and Logistic Regression are tuned with `GridSearchCV`; the final model scores every \*\*active\*\* employee with an attrition probability.

5\. \*\*Reporting:\*\* predictions are written back to MySQL (`predicted\_dataset`) and visualised in Power BI.



\---



\## Key Findings



\### Historical attrition (all 1,470 employees)



| Metric | Value |

|---|---|

| Total employees | 1,470 (1,233 active) |

| Total attrition | 237 (\*\*16.1%\*\*) |

| Average tenure | 7.01 years (4.23 in current role) |

| Average monthly income | $6,503 |



\- \*\*Sales Representatives\*\* have by far the highest attrition rate of any role (\*\*39.76%\*\*), followed by Laboratory Technicians (23.94%) and Human Resources (23.08%).

\- \*\*Sales\*\* is the department with the highest attrition rate (\*\*20.63%\*\*), then Human Resources (19.05%) and Research \& Development (13.84%).

\- \*\*Low pay is a strong signal:\*\* employees earning under $3K/month leave at \*\*28.61%\*\*, versus 8.90% for those earning above $10K.

\- \*\*Job satisfaction matters:\*\* employees with the lowest satisfaction leave at \*\*22.84%\*\*, about double the rate of the "Very High" group (11.33%).

\- \*\*Income difference is statistically significant:\*\* average monthly income was \*\*$6,833 for stayers vs. $4,787 for leavers\*\* (Welch's t-test, t = 7.48, p ≈ 4.4e-13).



\### Predicted attrition (1,233 active employees)



| Metric | Value |

|---|---|

| Employees flagged as likely to leave | 212 |

| Predicted attrition rate | \*\*17.19%\*\* |

| Average attrition probability | 21.3% |



\- Predicted risk is concentrated in \*\*Laboratory Technicians (60)\*\*, \*\*Sales Executives (49)\*\* and \*\*Research Scientists (36)\*\*.

\- \*\*Research \& Development\*\* holds the most at-risk employees in absolute terms (123), followed by Sales (77) and HR (12).

\- About \*\*two-thirds (66%)\*\* of flagged employees work overtime.

\- Attrition probability is highest for employees with \*\*short tenure\*\* (roughly the first 10 years) and drops for long-serving staff.



\---



\## Machine Learning Details



\*\*Target:\*\* `Attrition` (Yes = 1, No = 0)



\*\*Preprocessing\*\*

\- Dropped non-informative columns: `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`.

\- \*\*`Gender` was excluded from the model\*\* to avoid using a protected attribute as a predictor.

\- Encoded `OverTime` (binary) and `Education` (ordinal); one-hot encoded remaining categoricals (`drop\_first=True`).

\- Standardised numeric features with `StandardScaler` (fit on train only).

\- 80/20 train/test split (`random\_state=42`).



\*\*Models compared\*\*



| Model | Tuning | Notes |

|---|---|---|

| Random Forest | `GridSearchCV`, 5-fold, scoring = recall | Accuracy 0.87 but recall for leavers only \*\*0.10\*\* |

| Logistic Regression | `GridSearchCV`, 5-fold, scoring = F1 | Best params: `C=1`, `penalty=l1`, `solver=liblinear` |



\*\*Final model: Logistic Regression with a 0.40 decision threshold.\*\* The dataset is imbalanced (about 16% leavers), so the default 0.50 cut-off misses too many leavers. Lowering it trades some precision for better recall:



| Threshold | Precision (Yes) | Recall (Yes) | F1 (Yes) |

|---|---|---|---|

| 0.50 | 0.56 | 0.36 | 0.44 |

| 0.45 | 0.50 | 0.41 | 0.45 |

| \*\*0.40\*\* | \*\*0.50\*\* | \*\*0.49\*\* | \*\*0.49\*\* |



Confusion matrix at 0.40 (test set, n = 294):



|  | Predicted stay | Predicted leave |

|---|---|---|

| \*\*Actual stay\*\* | 227 | 28 |

| \*\*Actual leave\*\* | 20 | 19 |



\*\*Scoring:\*\* the fitted model scores only employees with `Attrition = No` (the active workforce). Output columns `Attrition\_probability` (%) and `Predict\_attrition` are saved to the `predicted\_dataset` table, which feeds the second dashboard page.



\### Limitations

\- The model finds about half of true leavers at \~50% precision. Treat scores as a \*\*prioritisation aid for HR conversations\*\*, not as individual verdicts.

\- The test set is small (39 leavers), so metrics are noisy.

\- The dataset is a \*\*synthetic sample\*\* from IBM; conclusions may not transfer to a real organisation.

\- Correlation is not causation: features like overtime and low income are associated with leaving, but this analysis does not prove they cause it.



\---



\## Repository Structure



```

├── main.ipynb            # SQL + EDA + hypothesis test + ML pipeline

├── hr\_analytics.pbix     # Power BI dashboard (2 pages)

├── images/

│   ├── attrition\_overview.png

│   └── predicted\_attrition.png

└── README.md

```



\---



\## Tech Stack



\- \*\*Database:\*\* MySQL (via SQLAlchemy + PyMySQL)

\- \*\*Analysis:\*\* Python, pandas, NumPy, SciPy

\- \*\*Visualisation:\*\* matplotlib, seaborn, Power BI

\- \*\*Machine learning:\*\* scikit-learn (Logistic Regression, Random Forest, GridSearchCV)



\---



\## Getting Started



\### 1. Prerequisites

\- Python 3.9+

\- MySQL Server

\- Power BI Desktop (to open the `.pbix` file)



```bash

pip install pandas numpy matplotlib seaborn scipy scikit-learn sqlalchemy pymysql jupyter

```



\### 2. Set up the database

1\. Download the IBM HR Analytics CSV from Kaggle.

2\. Create a database named `hr` and load the CSV into a table called `hr\_attrition`.



\### 3. Configure the connection

The notebook connects with SQLAlchemy. Keep credentials out of source control by reading them from environment variables:



```python

import os

from sqlalchemy import create\_engine



engine = create\_engine(

&#x20;   f"mysql+pymysql://{os.environ\['DB\_USER']}:{os.environ\['DB\_PASSWORD']}@localhost/hr"

)

```



\### 4. Run the notebook

```bash

jupyter notebook main.ipynb

```

Running it end to end will (re)create the `predicted\_dataset` table in MySQL.



\### 5. Open the dashboard

Open `hr\_analytics.pbix` in Power BI Desktop and refresh the data source so it points at your local MySQL instance.



\---



\## Possible Next Steps

\- Handle class imbalance with SMOTE or class weights and compare against gradient boosting (XGBoost / LightGBM).

\- Add model explainability (SHAP or logistic coefficients) to show \*why\* each employee is flagged.

\- Calibrate probabilities and validate with cross-validated threshold selection.

\- Add a retention-cost estimate to prioritise the highest-value at-risk employees.



\---



\## Acknowledgements

Dataset: IBM HR Analytics Employee Attrition \& Performance (Kaggle sample data).

