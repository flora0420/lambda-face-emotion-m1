from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from deeplab import DeepLabModel
from mangum import Mangum
import numpy as np
import cv2
import io

model_path = './frozen_inference_graph.pb'
model = DeepLabModel(model_path)

app = FastAPI(title='Serverless Lambda FastAPI') #, root_path="/Prod/")


@app.post("/face-bokeh/{query}", tags=["Face Bokeh"])
async def bokeh(file: UploadFile = File(...), query: str = ''):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    mask = model.get_mask(img)
    return_img = model.transform(img, mask, query)
    _, png_img = cv2.imencode('.PNG', return_img)
    return StreamingResponse(io.BytesIO(png_img.tobytes()), media_type="image/png")


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Ok"}


handler = Mangum(app=app)