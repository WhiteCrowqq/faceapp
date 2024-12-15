from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import cv2
import shutil
import numpy as np
import logging
import uuid
from PIL import Image
from insightface.app import FaceAnalysis
import insightface
from gfpgan import GFPGANer

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Configure upload and result folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HERO_FOLDER = 'hero'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HERO_FOLDER, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load face swapper and GFPGAN
swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)


def simple_face_swap(source_image, target_image, face_app, swapper):
    logging.info("Starting face swap...")
    faces1 = face_app.get(source_image)
    faces2 = face_app.get(target_image)

    if len(faces1) == 0 or len(faces2) == 0:
        return None  # No faces detected

    face1 = faces1[0]
    face2 = faces2[0]

    swapped_image = swapper.get(source_image, face1, face2, paste_back=True)
    logging.info("Face swap completed.")
    return swapped_image


def enhance_face(image):
    logging.info("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)

    if isinstance(restored_img, Image.Image):
        restored_img = np.array(restored_img)
    if isinstance(restored_img, np.ndarray):
        logging.info("Face enhancement completed.")
        return restored_img
    else:
        raise ValueError("Enhanced image is not a valid numpy array")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the HTML page for the index."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, targetImage: UploadFile = File(...)):
    """Handle file upload and perform face swapping."""
    # Save the uploaded target image
    target_image_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)
    with open(target_image_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    # Randomly pick a source image from the HERO_FOLDER
    hero_images = [
        os.path.join(HERO_FOLDER, f) for f in os.listdir(HERO_FOLDER)
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    if not hero_images:
        raise HTTPException(status_code=500, detail="No source images found in hero folder")

    source_image_path = np.random.choice(hero_images)

    # Read images with OpenCV
    source_image = cv2.imread(source_image_path)
    target_image_cv = cv2.imread(target_image_path)

    if source_image is None or target_image_cv is None:
        raise HTTPException(status_code=500, detail="Failed to read images")

    # Perform face swap
    swapped_image = simple_face_swap(source_image, target_image_cv, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    # Enhance the swapped image
    enhanced_image = enhance_face(swapped_image)

    # Save the result image
    result_filename = f"swap_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, enhanced_image)

    # Redirect to result page
    return templates.TemplateResponse("result.html", {"request": request, "filename": result_filename})


@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Serve the resulting swapped image."""
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
