
import pandas as pd
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder
)
from sklearn import set_config

set_config(transform_output="pandas")

# columns
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = ["weather","type_of_order","type_of_vehicle","festival","city_type","is_weekend","order_time_of_day"]

ordinal_cat_cols = ["traffic", "distance_type"]

target_col = "time_taken"

# ordinal categories
traffic_order = ["low", "medium", "high", "jam"]
distance_type_order = ["short", "medium", "long", "very_long"]


def load_data(data_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(data_path)
        print("Data loaded successfully")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame | None:
    try:
        print("Original dataset shape:", data.shape)

        df = data.dropna()

        print("Dataset shape after dropping missing values:", df.shape)
        return df

    except Exception as e:
        print(f"Error dropping missing values: {e}")
        return None


def create_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(), num_cols),
            (
                "nominal_encode",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                nominal_cat_cols
            ),
            (
                "ordinal_encode",
                OrdinalEncoder(
                    categories=[
                        traffic_order,
                        distance_type_order
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ordinal_cat_cols
            )
        ],
        remainder="passthrough",
        n_jobs=-1,
        verbose_feature_names_out=False
    )

    return preprocessor


def join_X_and_y( X: pd.DataFrame, y: pd.Series ) -> pd.DataFrame:
    return X.join(y, how="inner")


def save_data(data: pd.DataFrame, save_path: str) -> pd.DataFrame | None:
    try:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        data.to_csv(save_path, index=False)
        print(f"Data saved successfully: {save_path}")

        return data

    except Exception as e:
        print(f"Error saving data: {e}")
        return None


def save_transformer(transformer, save_path: str) -> bool:
    try:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(transformer, f)

        print(f"Preprocessor saved successfully: {save_path}")
        return True

    except Exception as e:
        print(f"Error saving preprocessor: {e}")
        return False


if __name__ == "__main__":

    # paths
    train_data_path = os.path.join("data", "interim", "train.csv")
    test_data_path = os.path.join("data", "interim", "test.csv")

    train_trans_path = os.path.join(
        "data", "processed", "train_trans.csv"
    )
    test_trans_path = os.path.join(
        "data", "processed", "test_trans.csv"
    )
    transformer_path = os.path.join(
        "models", "preprocessor.joblib"
    )

    # load train and test data
    train_df = load_data(train_data_path)
    test_df = load_data(test_data_path)

    if train_df is not None and test_df is not None:

        # drop missing values
        train_df = drop_missing_values(train_df)
        test_df = drop_missing_values(test_df)

        if train_df is not None and test_df is not None:

            # split X and y
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # create preprocessor
            preprocessor = create_preprocessor()

            # fit and transform training data
            X_train_trans = preprocessor.fit_transform(X_train)
            print("Training data transformed successfully")

            # transform testing data only
            X_test_trans = preprocessor.transform(X_test)
            print("Testing data transformed successfully")

            # join X and y
            train_trans_df = join_X_and_y(
                X_train_trans, y_train
            )
            test_trans_df = join_X_and_y(
                X_test_trans, y_test
            )

            # save transformed data
            save_data(train_trans_df, train_trans_path)
            save_data(test_trans_df, test_trans_path)

            # save fitted preprocessor
            save_transformer(preprocessor, transformer_path)
