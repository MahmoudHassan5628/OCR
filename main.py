from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from easyocr import Reader
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Initialize the EasyOCR reader
reader = Reader(['en'], gpu=False)

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    # Convert the image to grayscale
    # image = image.convert('L')

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Use EasyOCR to read the text with details
    result = reader.readtext(image_np, detail=1)

    # Find the text with the highest confidence
    if result:
        highest_confidence_text = max(result, key=lambda x: x[2])
        extracted_text = highest_confidence_text[1]
    else:
        extracted_text = ""

    # Return the result as JSON
    return JSONResponse(content={"extracted_text": extracted_text})

@app.get("/")
async def root():
    return {"message": "Welcome to the EasyOCR FastAPI service. Use the /extract-text endpoint to extract text from an image."}
