from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
import cv2
import numpy as np
import os
import pickle
from ultralytics import YOLO

model = YOLO('best.pt') 
app = FastAPI()

@app.get("/")
async def read_root():
    return {'message': 'Hello World!'}

@app.post("/detect/")
async def detect_objects(file: UploadFile):

    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()

    # move the cursor back to the beginning
    await file.seek(0)

    if file_size > 2 * 1024 * 1024:
        # more than 2 MB
        raise HTTPException(status_code=400, detail="File too large. Should be less than 2MB")

    # check the content type (MIME type)
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Should be jpg or png")

    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    detections = model.predict(image, save=True, conf=0.1)
    path = os.path.join(detections[0].save_dir, os.listdir(detections[0].save_dir)[-1])


    return FileResponse(path)
