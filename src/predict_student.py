import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "dataset/student-mat.csv",
    sep=";"
)


# =========================================================
# 2. CREATE TARGET
# =========================================================

df["Risk"] = (df["G3"] < 10).astype(int)


# =========================================================
# 3. USE SAME 9 FEATURES
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


# =========================================================
# 4. PREPROCESSING
# =========================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
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
# 5. TRAIN RANDOM FOREST
# =========================================================

model = Pipeline(
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

model.fit(
    X,
    y
)

print("\nModel trained successfully.")


# =========================================================
# 6. NEW STUDENT DETAILS
# =========================================================

print("\n========================================")
print("NEW STUDENT RISK PREDICTION")
print("========================================")

print("\nEnter student details:\n")


student = {}


student["age"] = int(
    input("Age: ")
)

student["studytime"] = int(
    input("Study time (1-4): ")
)

student["failures"] = int(
    input("Previous failures (0-3): ")
)

student["absences"] = int(
    input("Number of absences: ")
)

student["schoolsup"] = input(
    "School support (yes/no): "
).lower()

student["famsup"] = input(
    "Family support (yes/no): "
).lower()

student["higher"] = input(
    "Wants higher education (yes/no): "
).lower()

student["G1"] = int(
    input("First period grade G1 (0-20): ")
)

student["G2"] = int(
    input("Second period grade G2 (0-20): ")
)


# =========================================================
# 7. CREATE NEW STUDENT DATAFRAME
# =========================================================

student_data = pd.DataFrame(
    [student]
)


# =========================================================
# 8. PREDICT
# =========================================================

prediction = model.predict(
    student_data
)[0]

risk_probability = model.predict_proba(
    student_data
)[0][1]


# =========================================================
# 9. DISPLAY RESULT
# =========================================================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")


if prediction == 1:

    print("Risk Status      : AT RISK")

else:

    print("Risk Status      : NOT AT RISK")


print(
    "Risk Probability :",
    round(
        risk_probability * 100,
        2
    ),
    "%"
)

print("========================================")