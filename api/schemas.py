
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DeliveryPredictionInput(BaseModel):

    rider_age: Annotated[
        float, Field(..., ge=18, le=70, description="Age of the delivery rider in years")
    ]
    rider_ratings: Annotated[
        float, Field(..., ge=0, le=5, description="Customer rating of the delivery rider")
    ]

    weather: Annotated[
        Literal["fog", "sandstorms", "stormy", "sunny", "windy"],
        Field(..., description="Weather conditions during delivery")
    ]

    traffic: Annotated[
        Literal["low", "medium", "high", "jam"],
        Field(..., description="Traffic conditions on the delivery route")
    ]

    vehicle_condition: Annotated[
        int, Field(..., ge=0, le=3, description="Condition rating of the delivery vehicle")
    ]

    type_of_order: Annotated[
        Literal["drinks", "meal", "snack", "buffet"],
        Field(..., description="Category of the customer's order")
    ]

    type_of_vehicle: Annotated[
        Literal["motorcycle", "scooter", "electric_scooter", "bicycle"],
        Field(..., description="Type of vehicle used for delivery")
    ]

    multiple_deliveries: Annotated[
        int, Field(..., ge=0, le=3, description="Number of deliveries handled together")
    ]

    festival: Annotated[
        Literal["yes", "no"],
        Field(..., description="Whether a festival is occurring")
    ]

    city_type: Annotated[
        Literal["semi-urban", "urban", "metropolitian"],
        Field(..., description="Classification of the delivery city")
    ]

    is_weekend: Annotated[
        int, Field(..., ge=0, le=1, description="Whether the order is placed on a weekend: 1=yes, 0=no")
    ]

    pickup_time_minutes: Annotated[
        float, Field(..., ge=0, description="Time taken to pick up the order, in minutes")
    ]

    order_time_of_day: Annotated[
        Literal["afternoon", "evening", "morning", "night"],
        Field(..., description="Time of day when the order is placed")
    ]

    distance_km: Annotated[
        float, Field(..., ge=0, description="Distance to the delivery destination, in kilometers")
    ]

    distance_type: Annotated[
        Literal["short", "medium", "long", "very_long"],
        Field(..., description="Category representing the delivery distance")
    ]


class DeliveryPredictionResponse(BaseModel):

    predicted_time_taken: Annotated[
        float,
        Field(..., description="Predicted delivery time in minutes")
    ]
