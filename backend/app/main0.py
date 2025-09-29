import numpy as np
import cv2

from fastapi import FastAPI, WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse
from app.services import Detect
import base64


app = FastAPI()
detector = Detect()

# --------------------------------API--------------------
# websocket connection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # remove "data:image/jpeg;base64," header if present
        if data.startswith("data:image"):
            data = data.split(",")[1]

        # decode base64 to  bytes
        img_bytes = base64.b64decode(data)
        # convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # detetion 
        result = detector.detect(frame)
        await websocket.send_text(result)

