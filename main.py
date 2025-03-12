from fastapi import FastAPI
from google.cloud import aiplatform
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Vertex AI 클라이언트 초기화
prediction_client = aiplatform.gapic.PredictionServiceClient()

# 엔드포인트 구성: 실제 엔드포인트가 속한 프로젝트 ID 사용
PROJECT_ID = "learned-now-451904-n7"
LOCATION = "asia-northeast3"
ENDPOINT_ID = "5805856801261879296"
ENDPOINT = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"

# 입력 데이터 모델 정의 (모든 값은 문자열이어야 함)
class PredictRequest(BaseModel):
    instances: List[Dict[str, str]]

@app.post("/predict")
async def predict(request: PredictRequest):
    # Vertex AI는 문자열 타입을 기대하므로, "Time" 필드를 문자열로 변환
    for instance in request.instances:
        if "Time" in instance:
            instance["Time"] = str(instance["Time"])
    
    # Vertex AI 엔드포인트 호출
    response = prediction_client.predict(
        endpoint=ENDPOINT,
        instances=request.instances
    )

    # Vertex AI의 응답을 그대로 반환
    return {"predictions": response.predictions}
