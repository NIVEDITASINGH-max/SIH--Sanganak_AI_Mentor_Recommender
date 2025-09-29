# TEMP CHANGE
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# IMPORTANT: This imports the MentorRecommender class from the file recommender.py
from recommender import MentorRecommender 

# Initialize FastAPI app
app = FastAPI(
    title="AlmaLink AI Mentor Recommender API",
    description="An API to provide mentor recommendations to students based on their skills and goals using TF-IDF and Cosine Similarity.",
    version="1.0.0"
)

# --- API Data Models (using Pydantic) ---

class MenteeProfile(BaseModel):
    # Field examples are excellent for the Swagger UI
    skills: List[str] = Field(..., example=["Python", "Machine Learning", "Data Analysis", "React"])
    career_goals: str = Field(..., example="I want to become a data scientist in the fintech industry, focusing on predictive modeling.")
    industry_preference: str = Field(..., example="Fintech")

class RecommendationRequest(BaseModel):
    mentee_id: str = Field(..., example="student123")
    profile: MenteeProfile
    top_n: int = Field(5, gt=0, le=10, description="Number of recommendations to return, between 1 and 10.")

# Define the structure for the mentor details in the response
class MentorDetails(BaseModel):
    title: str
    skills: List[str]
    experience: str
    industry: str

# Define the structure for a single recommendation
class Recommendation(BaseModel):
    mentor_id: str
    name: str
    match_score: float = Field(..., description="Cosine Similarity score, ranging from 0.0 to 1.0.")
    details: MentorDetails

# Define the full response structure
class RecommendationResponse(BaseModel):
    mentee_id: str
    recommendations: List[Recommendation]

# --- API Logic ---

# Initialize the recommender system on startup.
# FIX: The path is corrected to look for 'mentors.csv' in the same directory.
try:
    recommender = MentorRecommender(profiles_csv_path="mentors.csv")
except Exception as e:
    # Handle critical startup failure (e.g., file not found, pandas error)
    # This prevents the app from starting if the model can't be initialized
    print(f"CRITICAL STARTUP ERROR: Failed to initialize recommender: {e}")
    recommender = None # Set to None to allow health check to fail gracefully

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint. Also checks if the recommender is loaded."""
    if recommender is None or recommender.df.empty:
        raise HTTPException(status_code=503, detail="API is running, but Recommender Model is not initialized. Check 'mentors.csv'.")
    return {"status": "API is running and Recommender Model is loaded."}

@app.post("/api/v1/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(request: RecommendationRequest):
    """
    Accepts a mentee's profile (skills, goals, industry) and returns a ranked list of suitable mentors.
    """
    if recommender is None or recommender.df.empty:
        raise HTTPException(status_code=503, detail="Recommender Model is unavailable. Cannot process request.")
        
    # Convert profile skills list to a comma-separated string for the recommender
    # (The recommender logic expects a single query string)
    mentee_profile_dict = {
        "skills": ", ".join(request.profile.skills),
        "career_goals": request.profile.career_goals,
        "industry_preference": request.profile.industry_preference
    }
    
    # Get recommendations from the recommender instance
    recommended_mentors = recommender.recommend(
        mentee_profile=mentee_profile_dict,
        top_n=request.top_n
    )
    
    # Check for internal recommender errors
    if isinstance(recommended_mentors, dict) and 'error' in recommended_mentors:
         raise HTTPException(status_code=500, detail=recommended_mentors['error'])
    
    # Format the response using our Pydantic models
    return RecommendationResponse(
        mentee_id=request.mentee_id,
        recommendations=recommended_mentors
    )

# To run this API (in your project folder):
# 1. Ensure you have installed dependencies: pip install -r requirements.txt
# 2. Run the command: uvicorn main:app --reload
# 3. Open your browser to http://127.0.0.1:8000/docs
