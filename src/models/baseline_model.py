import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error , r2_score
from sklearn.model_selection import cross_val_score
import mlflow
import dagshub
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone


import dagshub
dagshub.init(repo_owner='Pratik1017', repo_name='swiggy_delivery_time_prediction', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Pratik1017/swiggy_delivery_time_prediction.mlflow")

mlflow.set_experiment("Experiments of Baseline Models")



#load data
def train_data(url: str ) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        X_train = df.drop(columns=['time_taken'])
        y_train = df['time_taken']

        return X_train , y_train
    
    except Exception as e:
            print(f"Error: {e}")
            return None


def test_data(url: str ) -> pd.DataFrame:

    try:
        df = pd.read_csv(url)
        X_test = df.drop(columns=['time_taken'])
        y_test = df['time_taken']

        return X_test , y_test
    
    except Exception as e:
            print(f"Error: {e}")
            return None

    

def perform_cv(X_train: pd.DataFrame , y_train: pd.DataFrame ):

    try:
        models = [LinearRegression() , Ridge() , Lasso() , KNeighborsRegressor() , RandomForestRegressor() ,
          GradientBoostingRegressor() , XGBRegressor() , LGBMRegressor()]


        score_dict = {}
        best_mae = float("inf")
        best_model = None
        best_model_name = None

        with mlflow.start_run(run_name="Model_Comparison"):

            for model in models:

                model_name = type(model).__name__

                with mlflow.start_run(run_name=model_name, nested=True):

                    score = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=3,
                        scoring="neg_mean_absolute_error"
                    )

                    mae = -score.mean()
                    score_dict[model_name] = mae

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("mean_cv_mae", mae)

                    if mae < best_mae:
                        best_mae = mae
                        best_model = clone(model)
                        best_model_name = model_name

        return best_model

    except Exception as e:
                print(f"Error: {e}")
                return None


def predict(best_model, X_train, y_train, X_test, y_test):
    try:
        mlflow.set_experiment("Baseline Model Winner")

        with mlflow.start_run(run_name=best_model.__name__):

            model = best_model()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.log_metric("MAE", mae)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="best_model"
            )

            print(f"Model: {best_model.__name__}")
            print(f"MAE: {mae}")

            return mae

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":

      X_train, y_train = train_data("data/processed/train_trans.csv")
      X_test, y_test = test_data("data/processed/test_trans.csv")

      best_model = perform_cv(X_train , y_train)

      predict(best_model , X_train, y_train , X_test, y_test)


