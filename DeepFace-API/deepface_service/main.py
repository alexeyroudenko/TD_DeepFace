import os
from typing import Optional
import warnings
import base64
import glob
import pickle
import mimetypes
import numpy as np
from datetime import datetime


# Suppress TensorFlow warnings before importing DeepFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from deepface import DeepFace
from pydantic import BaseModel
import shutil

app = FastAPI(title="DeepFace API")

# Ensure model weights directory exists
weights_dir = "/root/.deepface/weights"
os.makedirs(weights_dir, exist_ok=True)

def copy_model_weights():
    """Copy model weights from mounted volume to DeepFace cache if needed."""
    # Model weight files that should be copied
    weight_files = [
        "age_model_weights.h5",
        "gender_model_weights.h5", 
        "facial_expression_model_weights.h5"
    ]
    
    print("Checking and copying model weights if needed...")
    for weight_file in weight_files:
        source_path = os.path.join(weights_dir, weight_file)
        
        if os.path.exists(source_path):
            print(f"✓ {weight_file} found in weights directory")
        else:
            print(f"⚠ Warning: {weight_file} not found in weights directory")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    copy_model_weights()
    print("DeepFace API startup completed")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request models
class VerifyRequest(BaseModel):
    image1: str
    image2: str

class AnalyzeRequest(BaseModel):
    image: str

class RecognizeRequest(BaseModel):
    image: str
    top_k: Optional[int] = 5    

@app.get("/")
async def root():
    return {"message": "DeepFace API is running", "status": "healthy"}

@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint working", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is operational"}

@app.get("/time")
async def get_server_time():
    """Get current server time in various formats"""
    now = datetime.now()
    utc_now = datetime.utcnow()
    
    # Get timezone info
    try:
        import time
        timezone_offset = time.timezone if time.daylight == 0 else time.altzone
        timezone_hours = -timezone_offset / 3600
        timezone_str = f"UTC{'+' if timezone_hours >= 0 else ''}{int(timezone_hours):02d}:00"
    except Exception:
        timezone_str = "UTC"
    
    return {
        "server_time": {
            "local": now.isoformat(),
            "utc": utc_now.isoformat(),
            "timestamp": now.timestamp(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone_str,
            "unix_timestamp": int(now.timestamp())
        }
    }

@app.post("/verify")
async def verify_faces(request: VerifyRequest):
    try:
        # Decode base64 images
        img1_data = base64.b64decode(request.image1)
        img2_data = base64.b64decode(request.image2)
        
        # Save temporary files
        temp_path1 = "/tmp/img1.jpg"
        temp_path2 = "/tmp/img2.jpg"
        
        with open(temp_path1, "wb") as f:
            f.write(img1_data)
        with open(temp_path2, "wb") as f:
            f.write(img2_data)
            
        # Verify faces
        result = DeepFace.verify(img1_path=temp_path1, img2_path=temp_path2)
        
        # Clean up temporary files
        os.remove(temp_path1)
        os.remove(temp_path2)
        
        return {"verified": result["verified"], "confidence": result["distance"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during face verification: {str(e)}")

@app.post("/analyze")
async def analyze_face(request: AnalyzeRequest):
    try:
        img_data = base64.b64decode(request.image)
        temp_path = "/tmp/analyze.jpg"
        
        with open(temp_path, "wb") as f:
            f.write(img_data)
        
        # Ensure models are available before analysis
        print(f"Starting face analysis... for image saved at {temp_path}")
        result = DeepFace.analyze(
            img_path=temp_path, 
            actions=['age', 'gender', 'emotion'],
            enforce_detection=True
        )
        
        # os.remove(temp_path)
        print("Face analysis completed successfully")
        return result
    except Exception as e:
        # Clean up temp file on error
        temp_path = "/tmp/analyze.jpg"
        # if os.path.exists(temp_path):
        #     os.remove(temp_path)
        
        error_msg = str(e)
        print(f"Analysis error: {error_msg}")
        
        # Provide more helpful error messages
        if "age_model_weights.h5" in error_msg:
            raise HTTPException(
                status_code=500, 
                detail="Face analysis model not available. Please rebuild the container to download required models."
            )
        elif "No face detected" in error_msg or "Face could not be detected" in error_msg:
            raise HTTPException(
                status_code=422, 
                detail="No face detected in the uploaded image. Please upload an image with a clear, visible face."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error during face analysis: {error_msg}")

@app.post("/recognize")
async def recognize_face(request: RecognizeRequest):
    # Ensure face database directory exists
    db_dir = os.path.join("./data", "faces")
    os.makedirs(db_dir, exist_ok=True)

    # Decode and save incoming image
    temp_path = "/tmp/recognize.jpg"
    try:
        img_data = base64.b64decode(request.image)
        with open(temp_path, "wb") as f:
            f.write(img_data)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    try:
        # DeepFace.find returns a DataFrame (or list of DataFrames if multiple faces)
        find_result = DeepFace.find(
            img_path=temp_path,
            db_path=db_dir,
            enforce_detection=True
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[recognize] DeepFace.find error: {error_msg}")

        # Handle known representation length mismatch by clearing cached representations once and retrying
        mismatch_indicators = [
            "Length of values",
            "does not match length of index",
        ]
        if any(ind in error_msg for ind in mismatch_indicators):
            try:
                rep_files = glob.glob(os.path.join(db_dir, "**", "representations_*.pkl"), recursive=True)
                for rf in rep_files:
                    try:
                        os.remove(rf)
                        print(f"[recognize] Removed stale representations file: {rf}")
                    except Exception as del_err:
                        print(f"[recognize] Failed to remove {rf}: {del_err}")

                # Manual fallback: compute embeddings and distances to avoid pandas index mismatch
                allowed_ext = (".jpg", ".jpeg", ".png", ".bmp")
                image_paths = [
                    p for p in glob.glob(os.path.join(db_dir, "**", "*"), recursive=True)
                    if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed_ext
                ]
                if not image_paths:
                    return {"matches": [], "message": "Face database is empty."}

                cache_path = os.path.join(db_dir, "embeddings_vggface_opencv.pkl")
                embeddings = None
                # Load cache if consistent
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "rb") as fh:
                            embeddings = pickle.load(fh)
                        if not isinstance(embeddings, list) or len(embeddings) != len(image_paths):
                            embeddings = None
                    except Exception as cache_err:
                        print(f"[recognize] Failed to load embeddings cache: {cache_err}")
                        embeddings = None

                if embeddings is None:
                    embeddings = []
                    for ip in image_paths:
                        try:
                            rep = DeepFace.represent(
                                img_path=ip,
                                model_name="VGG-Face",
                                detector_backend="opencv",
                                enforce_detection=False,
                                align=True
                            )
                            if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and "embedding" in rep[0]:
                                emb = np.array(rep[0]["embedding"], dtype=np.float32)
                            elif isinstance(rep, dict) and "embedding" in rep:
                                emb = np.array(rep["embedding"], dtype=np.float32)
                            else:
                                continue
                            embeddings.append((ip, emb))
                        except Exception as emb_err:
                            print(f"[recognize] Skip {ip}: {emb_err}")
                            continue
                    # Save cache if we have some embeddings
                    try:
                        with open(cache_path, "wb") as fh:
                            pickle.dump(embeddings, fh)
                    except Exception as save_err:
                        print(f"[recognize] Failed to save embeddings cache: {save_err}")

                # Represent query image
                try:
                    qrep = DeepFace.represent(
                        img_path=temp_path,
                        model_name="VGG-Face",
                        detector_backend="opencv",
                        enforce_detection=False,
                        align=True
                    )
                    if isinstance(qrep, list) and len(qrep) > 0 and isinstance(qrep[0], dict) and "embedding" in qrep[0]:
                        qemb = np.array(qrep[0]["embedding"], dtype=np.float32)
                    elif isinstance(qrep, dict) and "embedding" in qrep:
                        qemb = np.array(qrep["embedding"], dtype=np.float32)
                    else:
                        raise ValueError("Could not compute query embedding")
                except Exception as qerr:
                    raise HTTPException(status_code=422, detail=f"Could not compute embedding for captured frame: {qerr}")

                # Compute cosine distances
                def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
                    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                    if denom == 0:
                        return 1.0
                    return float(1.0 - float(np.dot(a, b) / denom))

                scored = [
                    (ip, cosine_distance(qemb, emb))
                    for ip, emb in embeddings
                ]
                scored.sort(key=lambda x: x[1])
                top_k = request.top_k or 5
                top = scored[:top_k]

                matches = []
                for ip, dist in top:
                    try:
                        label = os.path.splitext(os.path.basename(ip))[0]
                    except Exception:
                        label = str(ip)
                    
                    # Generate full URL for the image
                    image_url = f"http://localhost:8000/face_image?path={ip}"
                    
                    matches.append({
                        "label": label,
                        "identity": ip,
                        "image_url": image_url,
                        "distance": float(dist),
                        "distance_metric": "cosine_fallback"
                    })
                return {"matches": matches, "fallback": True}
            except Exception as retry_err:
                error_msg = str(retry_err)
                print(f"[recognize] Retry after clearing representations failed: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Recognition failed after cache rebuild: {error_msg}")
        else:
            if "Face could not be detected" in error_msg or "No face detected" in error_msg:
                raise HTTPException(status_code=422, detail="No face detected in the captured frame.")
            if "does not exist" in error_msg or "is an empty directory" in error_msg:
                # Treat empty DB as no matches rather than error
                return {"matches": [], "message": "Face database is empty."}
            raise HTTPException(status_code=500, detail=f"Error during face recognition: {error_msg}")
    finally:
        # Ensure temp file is removed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # Normalize result to a single DataFrame
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        pass

    if isinstance(find_result, list) and len(find_result) > 0:
        df = find_result[0]
    else:
        df = find_result

    # If no results (empty dataframe)
    if df is None or getattr(df, "empty", True):
        return {"matches": [], "message": "No matches found"}

    # Identify distance column (DeepFace uses model-based naming, e.g., VGG-Face_cosine)
    distance_col = None
    for col in df.columns:
        if col.endswith("_cosine") or col.endswith("_euclidean") or col.endswith("_euclidean_l2"):
            distance_col = col
            break

    # Fallbacks
    if distance_col is None:
        distance_col = "distance" if "distance" in df.columns else df.columns[-1]

    # Sort by ascending distance (lower is better)
    df_sorted = df.sort_values(by=distance_col, ascending=True)
    top_k = request.top_k or 5
    df_top = df_sorted.head(top_k)

    # Build response
    matches = []
    for _, row in df_top.iterrows():
        identity_path = row.get("identity", "")
        try:
            # Derive a label from file name (without extension)
            label = os.path.splitext(os.path.basename(identity_path))[0]
        except Exception:
            label = str(identity_path)
        try:
            distance_value = float(row.get(distance_col, 0.0))
        except Exception:
            # In rare cases value might be numpy types that fail casting; coerce via str->float
            distance_value = float(f"{row.get(distance_col, 0.0)}")
        
        # Generate full URL for the image
        image_url = f"http://localhost:8000/face_image?path={identity_path}"
        
        matches.append({
            "label": label,
            "identity": identity_path,
            "image_url": image_url,
            "distance": distance_value,
            "distance_metric": distance_col
        })

    return {"matches": matches}

@app.get("/face_image")
async def get_face_image(path: str = Query(..., description="Absolute path of an image under ./data/faces")):
    try:
        db_dir = os.path.abspath(os.path.join(".", "data", "faces"))
        # Support absolute identity paths from matches
        requested_path = os.path.abspath(path)
        if not requested_path.startswith(db_dir):
            raise HTTPException(status_code=400, detail="Invalid image path")
        if not os.path.isfile(requested_path):
            raise HTTPException(status_code=404, detail="Image not found")
        media_type, _ = mimetypes.guess_type(requested_path)
        return FileResponse(requested_path, media_type=media_type or "image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to serve image: {str(e)}")
    
@app.get("/img")
async def get_static_image():
    """Return a static demo image. Looks for demo.jpg in weights dir or returns a generated placeholder."""
    # Preferred path for a user-provided image
    candidate_paths = [
        os.path.join(weights_dir, "demo.jpg"),
        os.path.join(weights_dir, "demo.png"),
        "/tmp/analyze.jpg",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = f.read()
            # Infer mime type
            ext = os.path.splitext(path)[1].lower()
            media_type = 'image/png' if ext == '.png' else 'image/jpeg'
            return Response(content=data, media_type=media_type)

    # Fallback: tiny 1x1 transparent PNG
    transparent_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/er4n1kAAAAASUVORK5CYII="
    )
    return Response(content=base64.b64decode(transparent_png_base64), media_type="image/png")