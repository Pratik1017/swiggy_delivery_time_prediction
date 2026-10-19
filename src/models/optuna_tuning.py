
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import dagshub

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# Initialize DagsHub MLflow tracking
dagshub.init(
    repo_owner="Pratik1017",
    repo_name="swiggy_delivery_time_prediction",
    mlflow=True
)


X_train = None
y_train = None


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 800
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 70
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 20
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 100
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-4, 10, log=True
        ),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1
    }

    model = LGBMRegressor(**params)

    with mlflow.start_run(
        run_name=f"LightGBM_Trial_{trial.number}",
        nested=True
    ):

        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_param("model", "LightGBM")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )

        mean_mae = -scores.mean()
        std_mae = scores.std()

        mlflow.log_metric("mean_cv_mae", mean_mae)
        mlflow.log_metric("std_cv_mae", std_mae)

        print(
            f"Trial: {trial.number} | "
            f"Mean CV MAE: {mean_mae:.4f}"
        )

    return mean_mae


if __name__ == "__main__":

    # Data Loading
    train_data = pd.read_csv(
        "data/processed/train_trans.csv"
    )

    X_train = train_data.drop(columns=["time_taken"])
    y_train = train_data["time_taken"]

    # Create MLflow experiment
    mlflow.set_experiment(
        "Swiggy_LightGBM_Optuna_Tuning"
    )

    # Parent run
    with mlflow.start_run(
        run_name="Optuna_Optimization"
    ):

        study = optuna.create_study(
            direction="minimize"
        )

        study.optimize(
            objective,
            n_trials=50
        )

        print("\nBest CV MAE:", study.best_value)
        print("Best Parameters:", study.best_params)
        print("Best Trial:", study.best_trial.number)

        # Log best trial results
        mlflow.log_metric(
            "best_cv_mae",
            study.best_value
        )

        mlflow.log_param(
            "best_trial_number",
            study.best_trial.number
        )

        mlflow.log_params({
            f"best_{key}": value
            for key, value in study.best_params.items()
        })

        # Save best parameters
        best_params = pd.DataFrame(
            [study.best_params]
        )

        best_params.to_json(
            "best_params.json",
            orient="records",
            indent=4
        )

        mlflow.log_artifact(
            "best_params.json"
        )

        # Train the best model on full training data
        best_model = LGBMRegressor(
            **study.best_params,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

        best_model.fit(X_train, y_train)

        # Log best model in parent run
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_lightgbm_model"
        )

        print("\nBest model logged to MLflow!")
