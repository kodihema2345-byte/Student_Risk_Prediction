# Student Academic Risk Analysis and Prediction System

## Overview

The Student Academic Risk Analysis and Prediction System is a machine learning project that analyzes student academic data and predicts whether a student may be at academic risk.

The system uses historical student performance data to identify academic risk patterns and applies multiple machine learning classification algorithms for prediction.

## Features

* Student academic data analysis
* Academic risk classification
* Risk distribution analysis
* Absence and final-grade relationship analysis
* Comparison of multiple machine learning models
* Confusion matrix-based model evaluation
* New student academic risk prediction

## Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest

## Dataset

The project uses the UCI Student Performance dataset for Mathematics.

The target variable is created using the final grade:

* `G3 < 10` → At Risk
* `G3 >= 10` → Not At Risk

This threshold is used as the project's operational definition of academic risk.

## Features Used

The prediction models use the following features:

* Age
* Study Time
* Previous Failures
* Absences
* School Support
* Family Support
* Higher Education Intention
* First Period Grade (G1)
* Second Period Grade (G2)

The final grade (`G3`) is used to define the target and is not used as an input feature for prediction.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Project Structure

```text
Student-Risk-Analysis/
│
├── dataset/
├── src/
├── graphs/
├── output/
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the main analysis and model training:

```bash
python src/student_risk_analysis.py
```

Run prediction for a new student:

```bash
python src/predict_student.py
```

## Output

The project generates:

* Academic risk analysis graphs
* Model comparison results
* Confusion matrices
* Test-set predictions
* New student risk prediction

## Disclaimer

The prediction represents a machine-learning-based risk classification from the available student data. It should be treated as an academic early-warning aid rather than a guaranteed prediction of a student's future performance.
