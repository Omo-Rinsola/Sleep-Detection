import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, Field
from app.services import Detect

api = FastAPI()
detector = Detect()

# -----------------------------class-------------------


class SleepStatus(BaseModel):
    status: str = Field(..., description="sleep or awake status")

# --------------------------------API--------------------


@api.get("/")
def root():
    return {"message": " practice"}


@api.post("/sleepstatus", response_model=SleepStatus)
async def sleep_detection(file: UploadFile):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = detector.detect(frame)
        return SleepStatus(status=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")




