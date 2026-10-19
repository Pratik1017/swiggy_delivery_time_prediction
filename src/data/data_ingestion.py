
import pandas as pd
import numpy as np
import os

# Load Data
def load_data(url: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(url)
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Rename Columns
def change_column_names(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df.columns = df.columns.str.strip().str.lower()

        df.rename(columns={
        "delivery_person_id": "rider_id",
        "delivery_person_age": "rider_age",
        "delivery_person_ratings": "rider_ratings",
        "restaurant_latitude": "restaurant_latitude",
        "restaurant_longitude": "restaurant_longitude",
        "delivery_location_latitude": "delivery_latitude",
        "delivery_location_longitude": "delivery_longitude",
        "time_orderd": "order_time",
        "time_ordered": "order_time",
        "time_order_picked": "order_picked_time",
        "weatherconditions": "weather",
        "road_traffic_density": "traffic",
        "city": "city_type",
        "time_taken(min)": "time_taken"
        }, inplace=True)

        return df

    except Exception as e:
        print(f"Error renaming columns: {e}")
        return None


# Handle Missing Values
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = df.copy()

        df.replace(
            r"(?i)^\s*(nan|none|null)\s*$",
            np.nan,
            regex=True,
            inplace=True
        )

        return df

    except Exception as e:
        print(f"Error handling missing values: {e}")
        return None


# Clean Numeric Columns
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = df.copy()

        df.drop(columns=["id"], inplace=True, errors="ignore")

        df["rider_age"] = pd.to_numeric(
            df["rider_age"], errors="coerce"
        )

        df["rider_ratings"] = pd.to_numeric(
            df["rider_ratings"], errors="coerce"
        )

        df["multiple_deliveries"] = pd.to_numeric(
            df["multiple_deliveries"], errors="coerce"
        )

        df["time_taken"] = pd.to_numeric(
            df["time_taken"]
            .astype("string")
            .str.replace("(min)", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

        df = df[
            (df["rider_age"].isna() | (df["rider_age"] >= 18))
            & (df["rider_ratings"].isna() | (df["rider_ratings"] <= 5))
        ].copy()

        return df

    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None


# Extract City From Rider ID
def extract_city(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df["city_name"] = (
            df["rider_id"]
            .astype("string")
            .str.split("RES")
            .str[0]
        )

        return df

    except Exception as e:
        print(f"Error extracting city: {e}")
        return None


# Clean Categorical Columns
def clean_categorical_columns(
    df: pd.DataFrame
) -> pd.DataFrame | None:
    try:
        categorical_columns = [
            "weather",
            "traffic",
            "type_of_order",
            "type_of_vehicle",
            "festival",
            "city_type"
        ]

        for col in categorical_columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

        df["weather"] = df["weather"].str.replace(
            "conditions ",
            "",
            regex=False
        )

        return df

    except Exception as e:
        print(f"Error cleaning categorical columns: {e}")
        return None


# Extract Date Features
def extract_date_features(
    df: pd.DataFrame
) -> pd.DataFrame | None:
    try:
        df["order_date"] = pd.to_datetime(
            df["order_date"],
            dayfirst=True,
            errors="coerce"
        )

        df["order_day"] = df["order_date"].dt.day
        df["order_month"] = df["order_date"].dt.month

        df["order_day_of_week"] = (
            df["order_date"].dt.day_name().str.lower()
        )

        df["is_weekend"] = (
            df["order_date"].dt.dayofweek.isin([5, 6]).astype(float)
        )

        df.loc[df["order_date"].isna(), "is_weekend"] = np.nan

        return df

    except Exception as e:
        print(f"Error extracting date features: {e}")
        return None


# Extract Time Features
def extract_time_features(
    df: pd.DataFrame
) -> pd.DataFrame | None:
    try:
        order_time = pd.to_datetime(
            df["order_time"],
            format="mixed",
            errors="coerce"
        )

        picked_time = pd.to_datetime(
            df["order_picked_time"],
            format="mixed",
            errors="coerce"
        )

        df["order_time_hour"] = order_time.dt.hour

        pickup_minutes = (
            picked_time - order_time
        ).dt.total_seconds() / 60

        df["pickup_time_minutes"] = pickup_minutes.where(
            pickup_minutes >= 0,
            pickup_minutes + 1440
        )

        df["order_time_of_day"] = pd.cut(
            df["order_time_hour"],
            bins=[-1, 6, 12, 17, 20, 24],
            labels=[
                "after_midnight",
                "morning",
                "afternoon",
                "evening",
                "night"
            ],
            right=False
        )

        df.drop(
            columns=["order_time", "order_picked_time"],
            inplace=True
        )

        return df

    except Exception as e:
        print(f"Error extracting time features: {e}")
        return None


# Clean Latitude and Longitude
def clean_lat_long(
    df: pd.DataFrame,
    threshold: float = 1.0
) -> pd.DataFrame | None:
    try:
        coordinate_columns = [
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude"
        ]

        for col in coordinate_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            df.loc[df[col].abs() < threshold, col] = np.nan

        return df

    except Exception as e:
        print(f"Error cleaning coordinates: {e}")
        return None


# Calculate Haversine Distance
def calculate_haversine_distance(
    df: pd.DataFrame
) -> pd.DataFrame | None:
    try:
        lat1 = np.radians(df["restaurant_latitude"])
        lon1 = np.radians(df["restaurant_longitude"])
        lat2 = np.radians(df["delivery_latitude"])
        lon2 = np.radians(df["delivery_longitude"])

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(delta_lon / 2) ** 2
        )

        a = a.clip(0, 1)

        df["distance_km"] = (
            6371 * 2 * np.arcsin(np.sqrt(a))
        )

        return df

    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None


# Create Distance Categories
def create_distance_type(
    df: pd.DataFrame
) -> pd.DataFrame | None:
    try:
        df["distance_type"] = pd.cut(
            df["distance_km"],
            bins=[-np.inf, 5, 10, 15, np.inf],
            labels=["short", "medium", "long", "very_long"],
            right=False
        )

        return df

    except Exception as e:
        print(f"Error creating distance categories: {e}")
        return None


# Drop Unwanted Columns
def drop_columns(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame | None:
    try:
        df.drop(
            columns=columns,
            inplace=True,
            errors="ignore"
        )

        return df

    except Exception as e:
        print(f"Error dropping columns: {e}")
        return None


def perform_data_cleaning( df: pd.DataFrame ) -> pd.DataFrame | None:
    try:
        df = change_column_names(df)
        if df is None:
            return None

        df = handle_missing_values(df)
        if df is None:
            return None

        df = data_cleaning(df)
        if df is None:
            return None

        df = extract_city(df)
        if df is None:
            return None

        df = clean_categorical_columns(df)
        if df is None:
            return None

        df = extract_date_features(df)
        if df is None:
            return None

        df = extract_time_features(df)
        if df is None:
            return None

        df = clean_lat_long(df)
        if df is None:
            return None

        df = calculate_haversine_distance(df)
        if df is None:
            return None

        df = create_distance_type(df)
        if df is None:
            return None

        columns_to_drop = [
            "rider_id",
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
            "order_date",
            "order_time_hour",
            "order_day",
            "city_name",
            "order_day_of_week",
            "order_month"
        ]

        df = drop_columns(df, columns_to_drop)

        return df

    except Exception as e:
        print(f"Error in data cleaning pipeline: {e}")
        return None


#Main
if __name__ == "__main__":

    df = load_data("data/external/swiggy.csv")

    if df is not None:
        cleaned_df = perform_data_cleaning(df)

        if cleaned_df is not None:

            # Output Directory
            path_link = os.path.join("data", "raw")
            os.makedirs(path_link, exist_ok=True)

            output_path = os.path.join(
                path_link, "swiggy_cleaned.csv"
            )

            cleaned_df.to_csv(output_path, index=False)

            print("Data cleaning completed successfully!")