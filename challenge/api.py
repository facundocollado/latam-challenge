import os
import pickle
from typing import Any, Dict, List
import fastapi
import pandas as pd
import aiofiles
from challenge.model import DelayModel
from pydantic import BaseModel

app = fastapi.FastAPI()

model = None

# Load the model asynchronously
async def load_model():
    model_path = os.path.abspath("data/delay_model.pkl")
    async with aiofiles.open(model_path, 'rb') as file:
        model_data = await file.read()
    
    model = DelayModel()
    model._model = pickle.loads(model_data)
    

    return model

@app.on_event("startup")
async def startup_event():
    global model
    model = await load_model()


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
    
    load_model()  # Ensure model is loaded

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
    
    predictions = []

    try:
        predictions = model.predict(features=df)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise fastapi.HTTPException(status_code=500, detail="Internal Server Error")
                                    

    return {"predict": predictions}