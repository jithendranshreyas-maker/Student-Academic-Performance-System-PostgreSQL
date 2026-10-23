
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

students_df = pd.read_csv("students_5_years.csv")
curriculum_df = pd.read_csv("curriculum_5_years.csv")
academic_df = pd.read_csv("academic_records_5_years_rescaled.csv")
semester_df = pd.read_csv(
    "semester_results_5_years.csv",
    dtype={"Student_ID": str}
)
df = pd.read_csv("performance_results_rescaled.csv")

match_columns = [
    "subject_code",
    "quiz1",
    "quiz2",
    "midsem",
    "assignment",
    "attendance_percentage",
    "branch",
    "semester",
    "academic_year",
    "marks_obtained",
]

academic_lookup = academic_df[match_columns + ["student_id"]].copy()

for column in ["subject_code", "branch"]:
    df[column] = df[column].astype("string").str.strip().str.upper()
    academic_lookup[column] = (
        academic_lookup[column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

performance_with_ids = df.merge(
    academic_lookup,
    on=match_columns,
    how="left",
    validate="many_to_one",
    indicator=True,
)

print("Performance rows:", len(df))
print("Matched rows:", (performance_with_ids["_merge"] == "both").sum())
print("Unmatched rows:", (performance_with_ids["_merge"] == "left_only").sum())

if (performance_with_ids["_merge"] == "left_only").any():
    st.warning("Some performance records could not be matched. Check the data before training.")
    st.dataframe(
        performance_with_ids.loc[
            performance_with_ids["_merge"] == "left_only",
            match_columns
        ].head(10)
    )
    st.stop()

df = performance_with_ids.drop(columns="_merge")

print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())


print("Performance columns:")
print(df.columns.tolist())

print("\nAcademic record columns:")
print(academic_df.columns.tolist())

print("\nShared columns:")
print(sorted(set(df.columns) & set(academic_df.columns)))


# --------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# --------------------------------------------------
FEATURE_COLUMNS = [
    "subject_code",
    "quiz1",
    "quiz2",
    "midsem",
    "assignment",
    "attendance_percentage",
    "prev_sga",
    "prev_sem_percentage",
    "prev_total_backlogs",
    "branch",
    "semester",
]

TARGET_COLUMN = "marks_obtained"

missing_columns = [
    col for col in FEATURE_COLUMNS + [TARGET_COLUMN]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in performance_results_rescaled.csv: "
        f"{missing_columns}"
    )

# Student ID is used for splitting only, never as a model feature.
if "Student_ID" in df.columns:
    GROUP_COLUMN = "Student_ID"
elif "student_id" in df.columns:
    GROUP_COLUMN = "student_id"
else:
    raise ValueError(
        "A student ID column is required in performance_results_rescaled.csv "
        "for student-level train/test splitting. Add or reliably join the "
        "correct Student_ID column before training."
    )

X = df[FEATURE_COLUMNS].copy()
y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
groups = df[GROUP_COLUMN]

# Remove rows with missing targets or missing student IDs.
valid_rows = y.notna() & groups.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)
groups = groups.loc[valid_rows].astype(str).reset_index(drop=True)

# Convert numerical feature columns to numeric values.
NUMERIC_COLUMNS = [
    "quiz1",
    "quiz2",
    "midsem",
    "assignment",
    "attendance_percentage",
    "prev_sga",
    "prev_sem_percentage",
    "prev_total_backlogs",
    "semester",
]

for col in NUMERIC_COLUMNS:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Standardize categorical values and represent missing categories as NaN.
for col in ["subject_code", "branch"]:
    X[col] = X[col].astype("string").str.strip()
    X[col] = X[col].replace("", pd.NA).astype(object)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeature count:", len(FEATURE_COLUMNS))
print("Target:", TARGET_COLUMN)
print("Valid rows:", len(X))
print("Unique students:", groups.nunique())

print("\nTarget distribution:")
print(y.describe())

if y.empty:
    raise ValueError("No valid target values remain after cleaning.")

if groups.nunique() < 3:
    raise ValueError(
        "At least three distinct students are needed for the chosen "
        "student-level split and cross-validation."
    )


# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

cat_features = ["subject_code", "branch"]

num_features = [
    col for col in FEATURE_COLUMNS
    if col not in cat_features
]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features),
])


# --------------------------------------------------
# 4. STUDENT-LEVEL TRAIN/TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

if not set(groups_train).isdisjoint(set(groups_test)):
    raise RuntimeError(
        "Student leakage detected: a student appears in both train and test."
    )

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training students:", groups_train.nunique())
print("Testing students:", groups_test.nunique())


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    ),
}


# --------------------------------------------------
# 6. GROUP CROSS-VALIDATION
# --------------------------------------------------

n_splits = min(5, groups_train.nunique())

group_cv = GroupKFold(n_splits=n_splits)

results = []
fitted_models = {}

for model_name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        groups=groups_train,
        cv=group_cv,
        scoring="r2",
        n_jobs=1,
    )

    # Fit on the complete training set.
    pipeline.fit(X_train, y_train)

    # Evaluate only on students held out from training.
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "Test R2": r2,
        "Test MAE": mae,
        "Test RMSE": rmse,
        "Group CV Mean R2": cv_scores.mean(),
        "Group CV Std R2": cv_scores.std(),
    })

    fitted_models[model_name] = pipeline

    print(f"\n{model_name}")
    print("Group CV scores:", np.round(cv_scores, 4))
    print(f"Group CV mean R2: {cv_scores.mean():.4f}")
    print(f"Group CV std R2:  {cv_scores.std():.4f}")
    print(f"Test R2:          {r2:.4f}")
    print(f"Test MAE:         {mae:.4f}")
    print(f"Test RMSE:        {rmse:.4f}")


# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Rank using held-out MAE: lower is better.
results_df = results_df.sort_values(
    by="Test MAE",
    ascending=True,
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]

print("\nBest model by held-out MAE:", best_model_name)


# --------------------------------------------------
# 8. TEST-SET PREDICTION DIAGNOSTICS
# --------------------------------------------------

test_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual Marks": y_test.to_numpy(),
    "Predicted Marks": test_predictions,
})

prediction_results["Residual"] = (
    prediction_results["Actual Marks"]
    - prediction_results["Predicted Marks"]
)

prediction_results["Absolute Error"] = (
    prediction_results["Residual"].abs()
)

within_5 = (
    prediction_results["Absolute Error"].le(5).mean() * 100
)

within_10 = (
    prediction_results["Absolute Error"].le(10).mean() * 100
)

print("\nBEST MODEL TEST DIAGNOSTICS")
print(f"Within ±5 marks:  {within_5:.2f}%")
print(f"Within ±10 marks: {within_10:.2f}%")
print(
    "Mean residual:",
    round(prediction_results["Residual"].mean(), 4)
)
print(
    "Median absolute error:",
    round(prediction_results["Absolute Error"].median(), 4)
)

print("\nFirst 10 test predictions:")
print(prediction_results.head(10).to_string(index=False))


# --------------------------------------------------
# 9. SAVE RESULTS AND BEST MODEL
# --------------------------------------------------

results_df.to_csv("model_comparison_results.csv", index=False)
prediction_results.to_csv("best_model_test_predictions.csv", index=False)

joblib.dump(best_model, "best_marks_prediction_pipeline.pkl")

joblib.dump(FEATURE_COLUMNS, "model_feature_columns.pkl")

best_metrics = results_df.iloc[0].to_dict()

# Normalize metric names for consistent use in app2.py
best_metrics = {
    str(key).strip().lower(): value
    for key, value in best_metrics.items()
}

best_metrics["within_5_marks_percent"] = within_5
best_metrics["within_10_marks_percent"] = within_10
best_metrics["mean_training_marks"] = float(y_train.mean())

joblib.dump(best_metrics, "model_metrics.pkl")

print("\nSaved:")
print("- model_comparison_results.csv")
print("- best_model_test_predictions.csv")
print("- best_marks_prediction_pipeline.pkl")
print("- model_feature_columns.pkl")
print("- model_metrics.pkl")

("- model_comparison_results.csv")
("- best_model_test_predictions.csv")
("- best_marks_prediction_pipeline.pkl")
("- model_feature_columns.pkl")
("- model_metrics.pkl")