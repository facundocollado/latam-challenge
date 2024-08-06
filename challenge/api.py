import os
import pickle
from typing import Any, Dict, List
import fastapi
import pandas as pd
import aiofiles
from challenge.model import DelayModel
from pydantic import BaseModel
from contextlib import asynccontextmanager


model = None

# Load the model asynchronously
async def load_model() -> DelayModel:
    model_path = os.path.abspath("data/delay_model.pkl")
    async with aiofiles.open(model_path, 'rb') as file:
        model_data = await file.read()
    
    loaded_model = DelayModel()
    loaded_model._model = pickle.loads(model_data)

    return loaded_model


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("LIFESPAN ACTIVATED")
    global model
    model = await load_model()
    print(model._model)
    yield

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @property
    def is_valid_mes(self) -> bool:
        return 1 <= self.MES <= 12

class PredictionRequest(BaseModel):
    flights: List[Flight]


@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> Dict[str, Any]:

    global model
    
    # Model will be none only during test scenarios
    if model is None:
            model_path = os.path.abspath("data/delay_model.pkl")
            with open(model_path, 'rb') as file:
                model_data = file.read()
                model = DelayModel()
                model._model = pickle.loads(model_data)

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

    # Convert flights to DataFrame directly
    df = pd.DataFrame([flight.model_dump() for flight in request.flights])
    
    # Ensure MES is valid
    if not all(df['MES'].between(1, 12)):
        raise fastapi.HTTPException(status_code=400, detail="MES must be between 1 and 12")
    
    # Get dummies for features
    df = pd.get_dummies(df, columns=['OPERA', 'TIPOVUELO', 'MES'], drop_first=True)
    
    # Initialize DataFrame with the necessary columns
    df = df.reindex(columns=FEATURES_COLS, fill_value=0)
    
    # try:
    #     predictions = model.predict(features=df)

    # except Exception as e:
    #     print(f"Error occurred: {e}")
    #     raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")
                                    
    predictions = model.predict(features=df)

    return {"predict": predictions}