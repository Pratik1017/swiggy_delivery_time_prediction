
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Initialize DagsHub MLflow tracking
dagshub.init(
    repo_owner="Pratik1017",
    repo_name="swiggy_delivery_time_prediction",
    mlflow=True
)


def load_data():
    train_path = os.path.join(
        "data", "processed", "train_trans.csv"
    )
    test_path = os.path.join(
        "data", "processed", "test_trans.csv"
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=["time_taken"])
    y_train = train["time_taken"]

    X_test = test.drop(columns=["time_taken"])
    y_test = test["time_taken"]

    return X_train, y_train, X_test, y_test


def create_model():
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
        n_jobs=-1
    )

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Results")
    print("-" * 35)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return y_pred, mae, rmse, r2


def save_model(model):
    model_path = os.path.join("model.pkl")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(
        "\nModel saved successfully:",
        os.path.abspath(model_path)
    )

    return model_path


def main():
    # Load training and testing data
    X_train, y_train, X_test, y_test = load_data()

    # Create model
    model = create_model()

    # Set MLflow experiment
    mlflow.set_experiment(
        "Swiggy_Best_LightGBM_Model"
    )

    # Parent MLflow run
    with mlflow.start_run(
        run_name="Best_LightGBM_Final_Model"
    ):
        # Log hyperparameters
        mlflow.log_params(model.get_params())

        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred, mae, rmse, r2 = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Log test metrics
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # Save model in repository root
        model_path = save_model(model)

        # Log pickle file as MLflow artifact
        mlflow.log_artifact(model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
        sk_model=model,
        name="best_lightgbm_model",
        serialization_format="cloudpickle"
    )

        print("\nMLflow tracking completed!")
        print(
            "Run ID:",
            mlflow.active_run().info.run_id
        )


if __name__ == "__main__":
    main()
