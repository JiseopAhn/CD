from fastapi import FastAPI
from google.cloud import aiplatform
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

prediction_client = aiplatform.gapic.PredictionServiceClient()

PROJECT_ID = "learned-now-451904-n7"
LOCATION = "asia-northeast3"
ENDPOINT_ID = "5805856801261879296"
ENDPOINT = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"

@app.post("/predict")
async def predict(request: PredictRequest):
    
    response = prediction_client.predict(
        endpoint=ENDPOINT,
        instances=request.instances
    )

    return {"predictions": response.predictions}
