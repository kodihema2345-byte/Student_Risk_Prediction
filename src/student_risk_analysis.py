import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "dataset/student-mat.csv",
    sep=";"
)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================================
# 2. CREATE TARGET
# =========================================================

# G3 < 10  -> At Risk
# G3 >= 10 -> Not At Risk

df["Risk"] = (df["G3"] < 10).astype(int)

print("\nRisk Distribution:")
print(df["Risk"].value_counts())

print("\nRisk Meaning:")
print("0 = Not At Risk")
print("1 = At Risk")


# =========================================================
# 3. SELECT PRACTICAL FEATURES
# =========================================================

features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "higher",
    "G1",
    "G2"
]

X = df[features]
y = df["Risk"]

print("\nFeatures used by the ML models:")
print(features)


# =========================================================
# 4. BASIC DATA ANALYSIS
# =========================================================

print("\nAverage Feature Values by Risk Group:")

print(
    df.groupby("Risk")[
        [
            "age",
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "G3"
        ]
    ].mean()
)


# =========================================================
# 5. SAVE PROCESSED DATA
# =========================================================

processed_data = df[
    features + ["G3", "Risk"]
]

processed_data.to_csv(
    "output/student_risk_processed.csv",
    index=False
)

print("\nProcessed dataset saved.")


# =========================================================
# 6. GRAPH 1 - RISK DISTRIBUTION
# =========================================================

risk_counts = df["Risk"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Not At Risk", "At Risk"],
    [
        risk_counts.get(0, 0),
        risk_counts.get(1, 0)
    ]
)

plt.title("Student Academic Risk Distribution")
plt.xlabel("Risk Status")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    "graphs/risk_distribution.png"
)

plt.close()


# =========================================================
# 7. GRAPH 2 - FINAL GRADE DISTRIBUTION
# =========================================================

plt.figure(figsize=(7, 5))

plt.hist(
    df["G3"],
    bins=11
)

plt.title("Final Grade Distribution")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    "graphs/final_grade_distribution.png"
)

plt.close()


# =========================================================
# 8. GRAPH 3 - ABSENCES VS FINAL GRADE
# =========================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    df["absences"],
    df["G3"],
    alpha=0.6
)

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()

plt.savefig(
    "graphs/absences_vs_g3.png"
)

plt.close()


# =========================================================
# 9. GRAPH 4 - STUDY TIME VS ACADEMIC RISK
# =========================================================

study_risk = pd.crosstab(
    df["studytime"],
    df["Risk"]
)

study_risk.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Study Time vs Academic Risk")
plt.xlabel("Study Time Level")
plt.ylabel("Number of Students")

plt.legend(
    ["Not At Risk", "At Risk"],
    title="Risk"
)

plt.tight_layout()

plt.savefig(
    "graphs/study_time_vs_risk.png"
)

plt.close()


print("\n4 analysis graphs created successfully.")


# =========================================================
# 10. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 11. PREPROCESSING
# =========================================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# 12. LOGISTIC REGRESSION
# =========================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)

print("\n===== LOGISTIC REGRESSION =====")

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))


# =========================================================
# 13. DECISION TREE
# =========================================================

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

dt_precision = precision_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)

dt_recall = recall_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)

dt_f1 = f1_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)

print("\n===== DECISION TREE =====")

print("Accuracy :", round(dt_accuracy, 4))
print("Precision:", round(dt_precision, 4))
print("Recall   :", round(dt_recall, 4))
print("F1 Score :", round(dt_f1, 4))


# =========================================================
# 14. RANDOM FOREST
# =========================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_pred = random_forest_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

rf_precision = precision_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    random_forest_pred,
    zero_division=0
)

print("\n===== RANDOM FOREST =====")

print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))


# =========================================================
# 15. MODEL COMPARISON
# =========================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        dt_accuracy,
        rf_accuracy
    ],

    "Precision": [
        logistic_precision,
        dt_precision,
        rf_precision
    ],

    "Recall": [
        logistic_recall,
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        logistic_f1,
        dt_f1,
        rf_f1
    ]
})

print("\n===== MODEL COMPARISON =====")

print(comparison)


comparison.to_csv(
    "output/model_comparison.csv",
    index=False
)


# =========================================================
# 16. CONFUSION MATRICES
# =========================================================

models = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": decision_tree_pred,
    "Random Forest": random_forest_pred
}


for model_name, predictions in models.items():

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        f"\n{model_name} Confusion Matrix:"
    )

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not At Risk",
            "At Risk"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    plt.tight_layout()

    plt.savefig(
        f"graphs/confusion_matrix_{filename}.png"
    )

    plt.close()


# =========================================================
# 17. SAVE RANDOM FOREST TEST PREDICTIONS
# =========================================================

prediction_output = X_test.copy()

prediction_output["Actual_Risk"] = y_test

prediction_output["Predicted_Risk"] = (
    random_forest_pred
)

prediction_output.to_csv(
    "output/student_risk_predictions.csv",
    index=False
)


# =========================================================
# 18. FINAL MESSAGE
# =========================================================

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")

print("Dataset             : Student Performance")
print("Total Students      :", len(df))
print("Features Used       :", len(features))
print("Models Trained      : 3")
print("Target              : Academic Risk")
print("Risk Rule           : G3 < 10")

print("\nGenerated files:")
print("- Processed dataset")
print("- Model comparison")
print("- Test predictions")
print("- 4 analysis graphs")
print("- 3 confusion matrix graphs")

print("========================================")