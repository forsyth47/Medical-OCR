from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image

from schema import Prescription
from rx_parser import parse_prescription


app = FastAPI(title="Prescription OCR API")

# Allow cross-origin requests (useful for development and mobile/web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
async def root():
    """Returns a simple JSON response indicating the API is running."""
    return {"message": "Prescription OCR API is running."}


@app.post("/parse", response_model=Prescription)
async def parse_image(image: UploadFile = File(...)):
    """Accepts multipart/form-data with field name `image` and returns a Prescription JSON."""
    if not image:
        raise HTTPException(status_code=400, detail="no file uploaded")

    data = await image.read()

    # Validate that the uploaded bytes are an image (Pillow will raise on invalid files)
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")

    try:
        rx = parse_prescription(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return rx


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
