from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image

from schema import Prescription
from rx_parser import parse_prescription

ERROR_LOG_PATH = Path(__file__).resolve().parent / "error.txt"


def log_error(error: Exception, context: dict | None = None) -> None:
    """Appends full error details and traceback to error.txt with timestamp and divider."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "=" * 80

    lines = [
        divider,
        f"Timestamp: {timestamp}",
    ]

    if context:
        for key, value in context.items():
            lines.append(f"{key}: {value}")

    lines.extend([
        f"Error Type: {type(error).__name__}",
        f"Error Message: {str(error)}",
        "Traceback:",
        traceback.format_exc(),
        divider,
        "",  # extra newline separating subsequent error blocks
    ])

    log_entry = "\n".join(lines) + "\n"

    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)


app = FastAPI(title="Prescription OCR API")

# Allow cross-origin requests (useful for development and mobile/web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler for any unhandled exceptions to ensure they are logged."""
    log_error(
        exc,
        context={
            "Endpoint": f"{request.method} {request.url.path}",
            "Client": request.client.host if request.client else "unknown",
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
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
        log_error(
            e,
            context={
                "Endpoint": "POST /parse",
                "Uploaded Filename": image.filename,
                "Content-Type": image.content_type,
                "File Size (bytes)": len(data),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))

    return rx


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
