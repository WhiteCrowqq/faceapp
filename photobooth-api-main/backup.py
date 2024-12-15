from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
import shutil
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
import uuid
from fastapi import FastAPI, File, UploadFile

app = FastAPI()
# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
UPLOAD_CHAR = os.path.abspath(r"C:\Users\Anant\Music\photobooth-api-main\photobooth-api-main\hero")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_CHAR, exist_ok=True)

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)
    
    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected
    
    face1 = facesimg1[0]
    face2 = facesimg2[0]

    img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
    return img1_swapped

@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)
    with open(img2_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    sourceImage = cv2.imread(img1_path)
    targetImage = cv2.imread(img2_path)

    swapped_image = simple_face_swap(sourceImage, targetImage, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, swapped_image)

    return FileResponse(result_path)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), filename: str = Form(...)):
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Get file extension
    extension = os.path.splitext(file.filename)[1]
    
    # Ensure filename has the correct extension
    if not filename.endswith(extension):
        filename += extension
    
    file_location = os.path.join(UPLOAD_CHAR, filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    return JSONResponse(content={"filename": filename, "location": file_location})

# HTTP
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)