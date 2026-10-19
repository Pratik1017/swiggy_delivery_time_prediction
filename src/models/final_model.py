
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "interim" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "interim" / "test.csv"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
MODEL_PATH = ROOT_DIR / "model.pkl"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    target = "time_taken"

    if target not in train.columns or target not in test.columns:
        raise ValueError(
            f"Target column '{target}' must exist in both datasets."
        )

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Ensure training and test features have the same columns
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Train and test feature columns/order do not match."
        )

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, y_train, X_test, y_test


def load_preprocessor():
    with open(PREPROCESSOR_PATH, "rb") as file:
        preprocessor = pickle.load(file)

    print("Preprocessor loaded successfully.")
    return preprocessor


def create_pipeline(preprocessor):
    model = LGBMRegressor(
        n_estimators=197,
        learning_rate=0.03545159182046778,
        num_leaves=70,
        max_depth=17,
        min_child_samples=37,
        subsample=0.9935527627645536,
        subsample_freq=1,
        colsample_bytree=0.8515283081082318,
        reg_alpha=0.00017771855341790926,
        reg_lambda=0.2387434026492335,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results")
    print("-" * 35)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2


def save_model(pipeline):
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(pipeline, file)

    print("\nComplete pipeline saved at:", MODEL_PATH)
    return MODEL_PATH


def main():
    # Load raw data
    X_train, y_train, X_test, y_test = load_data()

    # Load the saved preprocessing transformer
    preprocessor = load_preprocessor()

    # Build preprocessing + LightGBM pipeline
    pipeline = create_pipeline(preprocessor)

    # Fit preprocessing and model on training data
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test data
    mae, rmse, r2 = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    # Save the complete pipeline for API inference
    model_path = save_model(pipeline)

    # Optional: log experiment to DagsHub / MLflow
    try:
        dagshub.init(
            repo_owner="Pratik1017",
            repo_name="swiggy_delivery_time_prediction",
            mlflow=True,
        )

        mlflow.set_experiment("Swiggy_Final_Pipeline")

        with mlflow.start_run(
            run_name="LightGBM_With_Preprocessor"
        ):
            mlflow.log_params(
                pipeline.named_steps["model"].get_params()
            )
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_artifact(str(model_path))

            # Cloudpickle avoids the default skops serialization path
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="swiggy_delivery_pipeline",
                serialization_format="cloudpickle",
            )

        print("MLflow logging completed.")

    except Exception as error:
        print("MLflow logging failed:", error)
        print("Local model is still saved at:", model_path)


if __name__ == "__main__":
    main()
