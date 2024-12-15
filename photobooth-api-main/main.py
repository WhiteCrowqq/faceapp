from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
import uuid
from gfpgan import GFPGANer
import numpy as np
import logging
from PIL import Image


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Initialize GFPGAN for face enhancement
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HERO_FOLDER = 'hero'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HERO_FOLDER, exist_ok=True)

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    logging.info("Starting face swap...")
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)

    logging.info(f"Number of faces detected in source image: {len(facesimg1)}")
    logging.info(f"Number of faces detected in target image: {len(facesimg2)}")

    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected

    face1 = facesimg1[0]
    face2 = facesimg2[0]

    img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)

    logging.info("Face swap completed.")
    return img1_swapped

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

@app.post("/api/swap-face/")
async def swap_faces(targetImage: UploadFile = File(...)):
    # Randomly pick a source image from the HERO_FOLDER
    hero_images = [os.path.join(HERO_FOLDER, f) for f in os.listdir(HERO_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not hero_images:
        raise HTTPException(status_code=500, detail="No source images found in hero folder")

    source_image_path = np.random.choice(hero_images)

    # Save the target image
    target_image_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)
    with open(target_image_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    # Read images with OpenCV
    sourceImage_cv = cv2.imread(source_image_path)
    targetImage_cv = cv2.imread(target_image_path)

    if sourceImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read source image with OpenCV: {source_image_path}")
    if targetImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read target image with OpenCV: {target_image_path}")

    logging.info(f"Source image shape: {sourceImage_cv.shape}")
    logging.info(f"Target image shape: {targetImage_cv.shape}")

    # Perform face swap
    swapped_image = simple_face_swap(sourceImage_cv, targetImage_cv, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    logging.info(f"Swapped image shape: {swapped_image.shape}")

    # Enhance the swapped image
    enhanced_image = enhance_face(swapped_image)

    logging.info(f"Enhanced image shape: {enhanced_image.shape}")

    # Save the result image with a numbered filename
    result_filename = f"swap_{len(os.listdir(RESULT_FOLDER)) + 1}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, enhanced_image)

    logging.info(f"Image saved to: {result_path}")

    return FileResponse(result_path)

# HTTP server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
