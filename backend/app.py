# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import sqlite3
import uuid
from datetime import datetime
import ollama
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="Car Insurance Validator Pro")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE = "insurance.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            hashed_password TEXT,
            full_name TEXT,
            created_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            vehicle_make TEXT,
            vehicle_model TEXT,
            vehicle_year INTEGER,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS submission_images (
            id TEXT PRIMARY KEY,
            submission_id TEXT,
            image_type TEXT,
            image_path TEXT,
            validation_result TEXT,
            validation_reason TEXT,
            FOREIGN KEY(submission_id) REFERENCES submissions(id)
        )
        """)

init_db()

# Models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class SubmissionCreate(BaseModel):
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int

class ImageUpload(BaseModel):
    submission_id: str
    image_type: str

# Auth setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMAGE_REQUIREMENTS = [
    {"type": "front", "label": "Front View", "description": "Clear view of the front of the vehicle"},
    {"type": "back", "label": "Rear View", "description": "Clear view of the rear of the vehicle"},
    {"type": "left", "label": "Left Side", "description": "Side view showing the left side of the vehicle"},
    {"type": "right", "label": "Right Side", "description": "Side view showing the right side of the vehicle"},
    {"type": "engine", "label": "Engine Bay", "description": "Clear view of the engine compartment"},
    {"type": "dashboard", "label": "Dashboard", "description": "Showing odometer reading"},
    {"type": "vin", "label": "VIN Plate", "description": "Clear photo of the vehicle identification number"},
    {"type": "registration", "label": "Registration", "description": "Current vehicle registration document"}
]

# Helper functions
def validate_image_with_ai(image_path: str, image_type: str) -> dict:
    """Validate the image using Ollama's vision model"""
    prompts = {
        "front": "Does this image clearly show the entire front of a vehicle? Check for: full view, no obstructions, good lighting.",
        "back": "Does this image clearly show the entire rear of a vehicle? Check for: full view, no obstructions, visible license plate.",
        "left": "Does this image show the complete left side of a vehicle? Check for: full side profile, no obstructions.",
        "right": "Does this image show the complete right side of a vehicle? Check for: full side profile, no obstructions.",
        "engine": "Does this image clearly show the engine compartment? Check for: clear view of engine components.",
        "dashboard": "Does this image show the vehicle dashboard with visible odometer reading? Check for: clear numbers.",
        "vin": "Does this image show a vehicle's VIN plate with readable characters? Check for: all characters legible.",
        "registration": "Does this image show a current vehicle registration document? Check for: readable text, current dates."
    }
    
    try:
        response = ollama.chat(
            model='llava:latest',
            messages=[{
                'role': 'user',
                'content': prompts[image_type],
                'images': [image_path]
            }]
        )
        
        content = response['message']['content'].lower().strip()
        
        return {
            "valid": "yes" in content or "valid" in content,
            "reason": content if "no" in content or "invalid" in content else "Valid image"
        }
    except Exception as e:
        return {"valid": False, "reason": f"Validation error: {str(e)}"}

def generate_pdf_report(submission_id: str):
    """Generate a PDF report for the submission"""
    with get_db() as conn:
        submission = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
        
        images = conn.execute(
            "SELECT * FROM submission_images WHERE submission_id = ?", (submission_id,)
        ).fetchall()
    
    filename = f"reports/{submission_id}.pdf"
    os.makedirs("reports", exist_ok=True)
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Vehicle Insurance Inspection Report")
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 100, f"Submission ID: {submission_id}")
    c.drawString(72, height - 120, f"Vehicle: {submission['vehicle_year']} {submission['vehicle_make']} {submission['vehicle_model']}")
    c.drawString(72, height - 140, f"Submission Date: {submission['created_at']}")
    
    # Images
    y_position = height - 180
    for img in images:
        if y_position < 200:
            c.showPage()
            y_position = height - 72
            
        try:
            img_reader = ImageReader(img['image_path'])
            c.drawImage(img_reader, 72, y_position - 120, width=200, height=150, preserveAspectRatio=True)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(300, y_position - 80, img['image_type'].capitalize())
            c.setFont("Helvetica", 10)
            c.drawString(300, y_position - 100, f"Status: {'Valid' if img['validation_result'] == 'yes' else 'Invalid'}")
            c.drawString(300, y_position - 120, f"Notes: {img['validation_reason']}")
            y_position -= 180
        except:
            continue
    
    c.save()
    return filename

# API Endpoints
@app.post("/api/submissions")
async def create_submission(data: SubmissionCreate, token: str = Depends(oauth2_scheme)):
    """Create a new vehicle submission"""
    submission_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    with get_db() as conn:
        conn.execute(
            "INSERT INTO submissions (id, user_id, vehicle_make, vehicle_model, vehicle_year, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (submission_id, token, data.vehicle_make, data.vehicle_model, data.vehicle_year, "pending", created_at)
        )
        conn.commit()
    
    return JSONResponse({
        "id": submission_id,
        "status": "pending",
        "required_images": IMAGE_REQUIREMENTS
    })

@app.post("/api/submissions/{submission_id}/upload")
async def upload_image(
    submission_id: str,
    image_type: str,
    file: UploadFile = File(...)
):
    """Upload and validate an image for a submission"""
    # Validate image type
    if not any(req['type'] == image_type for req in IMAGE_REQUIREMENTS):
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    # Save file
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{submission_id}_{image_type}{file_ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Validate with AI
    validation = validate_image_with_ai(filepath, image_type)
    
    # Save to database
    image_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO submission_images (id, submission_id, image_type, image_path, validation_result, validation_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, submission_id, image_type, filepath, 
             "yes" if validation['valid'] else "no", validation['reason'])
        )
        
        # Check if submission is complete
        images = conn.execute(
            "SELECT COUNT(*) as count FROM submission_images WHERE submission_id = ?", (submission_id,)
        ).fetchone()
        
        if images['count'] >= len(IMAGE_REQUIREMENTS):
            conn.execute(
                "UPDATE submissions SET status = 'complete' WHERE id = ?", (submission_id,)
            )
        
        conn.commit()
    
    return JSONResponse({
        "id": image_id,
        "valid": validation['valid'],
        "reason": validation['reason'],
        "image_url": f"/uploads/{filename}"
    })

@app.get("/api/submissions/{submission_id}/report")
async def get_submission_report(submission_id: str):
    """Generate and download a PDF report"""
    pdf_path = generate_pdf_report(submission_id)
    return FileResponse(pdf_path, filename=f"insurance_report_{submission_id}.pdf")

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)