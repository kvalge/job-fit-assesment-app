from fastapi import FastAPI, Form, Request, HTTPException
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
from datetime import datetime
from job_fit_analyzer import JobFitAnalyzer

app = FastAPI(title="Job Fit Assessment App", version="1.0.0")

os.makedirs("data", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

analyzer = JobFitAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save-cv")
async def save_cv(cv_text: str = Form(...)):
    """Save CV text to file"""
    try:
        cv_data = {
            "text": cv_text,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("data/cv.json", "w", encoding="utf-8") as f:
            json.dump(cv_data, f, ensure_ascii=False, indent=2)
        
        return {"message": "CV saved successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving CV: {str(e)}")

@app.post("/save-cover-letter")
async def save_cover_letter(cover_letter_text: str = Form(...)):
    """Save cover letter text to file"""
    try:
        cover_letter_data = {
            "text": cover_letter_text,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("data/cover_letter.json", "w", encoding="utf-8") as f:
            json.dump(cover_letter_data, f, ensure_ascii=False, indent=2)
        
        return {"message": "Cover letter saved successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving cover letter: {str(e)}")

@app.post("/analyze-job-fit")
async def analyze_job_fit(
    job_post_text: str = Form(...),
    skills_weight: Optional[float] = Form(None),
    experience_weight: Optional[float] = Form(None),
    overall_weight: Optional[float] = Form(None),
):
    """Analyze job fit based on CV, cover letter, and job post"""
    try:
        cv_text = ""
        cover_letter_text = ""
        
        if os.path.exists("data/cv.json"):
            with open("data/cv.json", "r", encoding="utf-8") as f:
                cv_data = json.load(f)
                cv_text = cv_data.get("text", "")
        
        if os.path.exists("data/cover_letter.json"):
            with open("data/cover_letter.json", "r", encoding="utf-8") as f:
                cover_letter_data = json.load(f)
                cover_letter_text = cover_letter_data.get("text", "")
        
        if not cv_text and not cover_letter_text:
            raise HTTPException(status_code=400, detail="No CV or cover letter found. Please save them first.")

        normalized_cv = analyzer.normalize_text(cv_text)
        normalized_cover = analyzer.normalize_text(cover_letter_text)
        normalized_job = analyzer.normalize_text(job_post_text)

        user_profile = "\n\n".join([p for p in [normalized_cv, normalized_cover] if p]).strip()

        section_weights = None
        if any(w is not None for w in (skills_weight, experience_weight, overall_weight)):
            section_weights = {
                "skills": float(skills_weight or analyzer.default_section_weights["skills"]),
                "experience": float(experience_weight or analyzer.default_section_weights["experience"]),
                "overall": float(overall_weight or analyzer.default_section_weights["overall"]),
            }

        analysis_result = await analyzer.analyze_fit(user_profile, normalized_job, section_weights=section_weights)
        
        return {
            "message": "Job fit analysis completed",
            "status": "success",
            "analysis": analysis_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing job fit: {str(e)}")

@app.get("/get-saved-data")
async def get_saved_data():
    """Get saved CV and cover letter data"""
    try:
        data = {"cv": "", "cover_letter": ""}
        
        if os.path.exists("data/cv.json"):
            with open("data/cv.json", "r", encoding="utf-8") as f:
                cv_data = json.load(f)
                data["cv"] = cv_data.get("text", "")
        
        if os.path.exists("data/cover_letter.json"):
            with open("data/cover_letter.json", "r", encoding="utf-8") as f:
                cover_letter_data = json.load(f)
                data["cover_letter"] = cover_letter_data.get("text", "")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading saved data: {str(e)}")

@app.post("/clear-all-data")
async def clear_all_data():
    """Clear all saved data and reset the application"""
    try:
        if os.path.exists("data/cv.json"):
            os.remove("data/cv.json")
        if os.path.exists("data/cover_letter.json"):
            os.remove("data/cover_letter.json")
        
        return {"message": "All data cleared successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)