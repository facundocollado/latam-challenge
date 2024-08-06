import os
import pickle
from typing import Any, Dict, List
import fastapi
import numpy as np
import pandas as pd
from challenge.model import DelayModel
from pydantic import BaseModel

app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    # @field_validator('MES')
    # def validate_mes(cls, value):
    #     if not (1 <= value <= 12):
    #         raise ValueError('MES must be between 1 and 12')
    #     return value
    @property
    def is_valid_mes(self) -> bool:
        return 1 <= self.MES <= 12

class PredictionRequest(BaseModel):
    flights: List[Flight]

@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> Dict[str, Any]:
    
    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    # Convert the values in data["flights"] into a key_value list
    key_value_list = []
    for flight in request.flights:
        if not flight.is_valid_mes:
            raise fastapi.HTTPException(status_code=400, detail="MES must be between 1 and 12")
        
        for key, value in flight.model_dump().items():
            key_value_list.append(f"{key}_{value}")

    # Initialize the DataFrame with FEATURES_COLS, setting all values to False
    df = pd.DataFrame(columns=FEATURES_COLS)
    row = {col: False for col in FEATURES_COLS}

    # Update the row based on key_value_list
    for kv in key_value_list:
        if kv in FEATURES_COLS:
            row[kv] = True  # Set to 1 or any desired value

    # Append the row to the DataFrame
    row_df = pd.DataFrame([row])
    df = pd.concat([df, row_df], ignore_index=True)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)


    model_path = os.path.abspath("data/delay_model.pkl")

    model = DelayModel()
    with open(model_path, 'rb') as file:
            model._model = pickle.load(file)

    prediction = model.predict(
        features=df
    )


    return {"predict": prediction}