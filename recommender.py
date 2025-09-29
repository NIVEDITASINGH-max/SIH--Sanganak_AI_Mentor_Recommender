# TEMP CHANGE
# recommender.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any

class MentorRecommender:
    """
    Handles data loading, TF-IDF model training, and the mentor recommendation logic.
    """
    def __init__(self, profiles_csv_path: str):
        self.profiles_csv_path = profiles_csv_path
        self.df = pd.DataFrame()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Load data and train model immediately on initialization
        self._load_data_and_train()
        
    def _load_data_and_train(self):
        print(f"Loading data from: {self.profiles_csv_path}")
        try:
            self.df = pd.read_csv(self.profiles_csv_path)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Data file not found at {self.profiles_csv_path}. Recommender unavailable.")
            return

        # Ensure required columns exist and fill NaNs
        required_cols = ['skills', 'experience', 'industry', 'mentor_id', 'name', 'title']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"CRITICAL ERROR: Column '{col}' missing from mentors.csv. Recommender unavailable.")
                self.df = pd.DataFrame()
                return
            self.df[col] = self.df[col].fillna('')
        
        # Combine features for matching
        # NOTE: Using a hypothetical 'shared_background' column for more comprehensive matching
        self.df['combined_features'] = (self.df['skills'] + ' ' + 
                                        self.df['experience'] + ' ' + 
                                        self.df['industry'] + ' ' + 
                                        self.df.get('shared_background', pd.Series([''] * len(self.df))).fillna(''))
        
        self._train_model()

    def _train_model(self):
        """Fits the TF-IDF vectorizer to the combined mentor features."""
        if self.df.empty:
            return

        # Fit the vectorizer and transform the mentor features
        try:
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
            print("TF-IDF Model Trained Successfully.")
        except Exception as e:
            print(f"ERROR: Failed to train TF-IDF model: {e}")
            self.tfidf_matrix = None


    def recommend(self, mentee_profile: Dict[str, str], top_n: int = 5) -> List[Dict[str, Any]] | Dict[str, str]:
        """
        Calculates similarity between the mentee profile and all mentor profiles.
        """
        if self.df.empty or self.tfidf_matrix is None:
            return {"error": "Recommender data or model matrix is empty."}

        # Combine mentee profile into a single query string
        mentee_query = f"{mentee_profile['skills']} {mentee_profile['career_goals']} {mentee_profile['industry_preference']}"

        # 1. Transform the mentee query using the FITTED vectorizer
        user_vec = self.tfidf.transform([mentee_query])

        # 2. Calculate similarity scores between mentee and all mentors
        sim_scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # 3. Get the indices of the top N mentors
        # np.argsort returns indices that would sort the array; [::-1] reverses it (highest score first)
        top_indices = sim_scores.argsort()[::-1][:top_n]

        # 4. Format the output
        recommended_mentors = []
        for i in top_indices:
            mentor = self.df.iloc[i]
            
            # Prepare skills list for the Pydantic model (must be a list of strings)
            skills_list = [s.strip() for s in mentor['skills'].split(',')] if mentor['skills'] else []

            recommended_mentors.append({
                "mentor_id": mentor['mentor_id'],
                "name": mentor['name'],
                # Match score is presented as a float between 0 and 1
                "match_score": round(sim_scores[i], 4),
                "details": {
                    "title": mentor['title'],
                    "skills": skills_list,
                    "experience": mentor['experience'],
                    "industry": mentor['industry']
                }
            })
        
        return recommended_mentors
