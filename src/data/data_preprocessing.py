import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

# load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["Data_Preparation"]["test_size"]
random_state = params["Data_Preparation"]["random_state"]


def load_data(data_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(data_path)
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def split_data( data: pd.DataFrame, test_size: float, random_state: int ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    try:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )

        return train_data, test_data

    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None


def export_train_data(train_data: pd.DataFrame) -> pd.DataFrame | None:
    try:
        path_link = os.path.join("data", "interim")
        os.makedirs(path_link, exist_ok=True)

        train_data.to_csv(
            os.path.join(path_link, "train.csv"),
            index=False
        )

        return train_data

    except Exception as e:
        print(f"Error exporting train data: {e}")
        return None


def export_test_data(test_data: pd.DataFrame) -> pd.DataFrame | None:
    try:
        path_link = os.path.join("data", "interim")
        os.makedirs(path_link, exist_ok=True)

        test_data.to_csv(
            os.path.join(path_link, "test.csv"),
            index=False
        )

        return test_data

    except Exception as e:
        print(f"Error exporting test data: {e}")
        return None


if __name__ == "__main__":

    data_path = os.path.join(
        "data", "cleaned", "swiggy_cleaned.csv"
    )

    df = load_data(data_path)

    if df is not None:
        train_data, test_data = split_data(
            df,
            test_size=test_size,
            random_state=random_state
        )

        if train_data is not None and test_data is not None:
            export_train_data(train_data)
            export_test_data(test_data)

            print("Done")