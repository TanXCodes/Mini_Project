"""Owner: Tamanna. Train Linear Regression + Random Forest, save best to models."""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from prepare_data import build_dataset


TARGET = "delay_min"

# These columns must NOT be used as input features.
# They would cause leakage.
LEAKAGE_COLUMNS = [
    "delay_min",
    "speed_kmph",
    "travel_time_min",
]


def main():

    # ---------------------------------------------------------
    # 1. Load prepared dataset
    # ---------------------------------------------------------
    train, validation, test = build_dataset()

    # Days 1-4 = training
    # Day 5 = testing
    train_data = pd.concat(
        [train, validation],
        ignore_index=True
    )

    test_data = test.copy()

    print("Training days:")
    print(sorted(train_data["day"].unique()))

    print("Test days:")
    print(sorted(test_data["day"].unique()))

    # ---------------------------------------------------------
    # 2. Select features
    # ---------------------------------------------------------
    feature_columns = [
        column
        for column in train_data.columns
        if column not in LEAKAGE_COLUMNS
    ]

    # segment_id is an identifier, not a useful ML feature
    feature_columns = [
        column
        for column in feature_columns
        if column != "segment_id"
    ]

    X_train = train_data[feature_columns].copy()
    y_train = train_data[TARGET].copy()

    X_test = test_data[feature_columns].copy()
    y_test = test_data[TARGET].copy()

    print("\nFeatures used:")
    print(feature_columns)

    print("\nLeakage check:")
    print("speed_kmph:", "speed_kmph" in feature_columns)
    print("travel_time_min:", "travel_time_min" in feature_columns)

    # ---------------------------------------------------------
    # 3. Identify categorical and numerical features
    # ---------------------------------------------------------
    categorical_features = [
        "time_slot",
        "road_class"
    ]

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    # ---------------------------------------------------------
    # 4. Preprocessing
    # ---------------------------------------------------------
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # ---------------------------------------------------------
    # 5. Models
    # ---------------------------------------------------------
    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression())
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    # ---------------------------------------------------------
    # 6. Train and evaluate
    # ---------------------------------------------------------
    results = []
    predictions = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = mean_squared_error(
            y_test,
            y_pred
        ) ** 0.5

        r2 = r2_score(
            y_test,
            y_pred
        )

        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R2   : {r2:.6f}")

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
        )

        predictions[model_name] = y_pred

    # ---------------------------------------------------------
    # 7. Save metrics
    # ---------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv(
        "outputs/model_metrics.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Select best model
    # ---------------------------------------------------------
    best_model_name = metrics_df.loc[
        metrics_df["RMSE"].idxmin(),
        "model"
    ]

    best_model = models[best_model_name]
    best_predictions = predictions[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # ---------------------------------------------------------
    # 9. Actual vs Predicted plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        best_predictions,
        alpha=0.3
    )

    min_value = min(
        y_test.min(),
        best_predictions.min()
    )

    max_value = max(
        y_test.max(),
        best_predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.title(
        f"Actual vs Predicted Delay - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/actual_vs_predicted.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # 10. Feature importance
    # ---------------------------------------------------------
    random_forest_pipeline = models["Random Forest"]

    rf_preprocessor = (
        random_forest_pipeline
        .named_steps["preprocessor"]
    )

    rf_model = (
        random_forest_pipeline
        .named_steps["model"]
    )

    feature_names = (
        rf_preprocessor
        .get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    # Plot top 15 features
    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # 11. Save best model
    # ---------------------------------------------------------
    joblib.dump(
        best_model,
        "models/delay_model.joblib"
    )

    print("\nSaved files:")
    print("outputs/model_metrics.csv")
    print("outputs/actual_vs_predicted.png")
    print("outputs/feature_importance.csv")
    print("outputs/feature_importance.png")
    print("models/delay_model.joblib")


if __name__ == "__main__":
    main()