import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Loads dataset, handles missing values, and removes duplicates."""
    df = pd.read_csv(csv_path)

    # Impute missing values
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates 5 engineered domain features."""
    df = df.copy()

    # 1. Study and Attendance Index
    df["Study_Attendance_Index"] = (df["Hours_Studied"] * df["Attendance"]) / 100

    # 2. Attendance Category
    df["Attendance_Category"] = pd.cut(
        df["Attendance"],
        bins=[0, 60, 75, 90, 100],
        labels=["Poor", "Average", "Good", "Excellent"],
    )

    # 3. Study Hours Category
    df["Study_Hours_Category"] = pd.cut(
        df["Hours_Studied"],
        bins=[0, 10, 20, 30, 100],
        labels=["Low", "Medium", "High", "Very High"],
    )

    # 4. Previous Performance Level
    df["Previous_Performance_Level"] = pd.cut(
        df["Previous_Scores"],
        bins=[0, 50, 70, 85, 100],
        labels=["Low", "Average", "Good", "Excellent"],
    )

    # 5. Academic Risk Score
    df["Academic_Risk_Score"] = 0
    df.loc[df["Attendance"] < 75, "Academic_Risk_Score"] += 1
    df.loc[df["Hours_Studied"] < 15, "Academic_Risk_Score"] += 1
    df.loc[df["Previous_Scores"] < 60, "Academic_Risk_Score"] += 1
    df.loc[df["Tutoring_Sessions"] == 0, "Academic_Risk_Score"] += 1
    df.loc[df["Motivation_Level"] == "Low", "Academic_Risk_Score"] += 1
    df.loc[df["Learning_Disabilities"] == "Yes", "Academic_Risk_Score"] += 1

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes all categorical variables to integers."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def evaluate_model(y_true, y_pred) -> dict:
    """Computes regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate():
    df = load_and_preprocess_data("data/student_performance.csv")
    df = engineer_features(df)
    df = encode_features(df)

    original_cols = [
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender",
    ]

    engineered_cols = original_cols + [
        "Study_Attendance_Index",
        "Attendance_Category",
        "Study_Hours_Category",
        "Previous_Performance_Level",
        "Academic_Risk_Score",
    ]

    y = df["Exam_Score"]

    # Model 1: Original Features
    X1 = df[original_cols]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X1, y, test_size=0.20, random_state=42
    )
    m1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        X_train1, y_train1
    )
    res1 = evaluate_model(y_test1, m1.predict(X_test1))

    # Model 2: Engineered Features
    X2 = df[engineered_cols]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y, test_size=0.20, random_state=42
    )
    m2 = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        X_train2, y_train2
    )
    res2 = evaluate_model(y_test2, m2.predict(X_test2))

    # Model 3: SelectKBest (k=10)
    selector = SelectKBest(score_func=f_regression, k=10)
    X3 = df[engineered_cols]
    selector.fit(X3, y)
    selected_cols = X3.columns[selector.get_support()]

    X_train3, X_test3, y_train3, y_test3 = train_test_split(
        df[selected_cols], y, test_size=0.20, random_state=42
    )
    m3 = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        X_train3, y_train3
    )
    res3 = evaluate_model(y_test3, m3.predict(X_test3))

    summary = pd.DataFrame(
        [
            {"Model": "Model 1 - Original Features", **res1},
            {"Model": "Model 2 - Engineered Features", **res2},
            {"Model": "Model 3 - Selected Features", **res3},
        ]
    )
    print("========== MODEL COMPARISON ==========")
    print(summary)


if __name__ == "__main__":
    train_and_evaluate()
