from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any

class JobFitAnalyzer:
    def __init__(self):
        """Initialize the job fit analyzer with pre-trained models"""
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from CV/cover letter text"""
        text = text.lower()
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'full_text': text
        }

        skill_keywords = ['skills', 'technical skills', 'competencies', 'abilities', 'technologies']
        exp_keywords = ['experience', 'work experience', 'employment', 'career', 'professional']
        edu_keywords = ['education', 'qualification', 'degree', 'university', 'college', 'certification']

        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()

            if any(keyword in paragraph_lower for keyword in skill_keywords):
                sections['skills'] += ' ' + paragraph
            elif any(keyword in paragraph_lower for keyword in exp_keywords):
                sections['experience'] += ' ' + paragraph
            elif any(keyword in paragraph_lower for keyword in edu_keywords):
                sections['education'] += ' ' + paragraph
        
        return sections
    
    def extract_job_requirements(self, job_post: str) -> Dict[str, str]:
        """Extract requirements and responsibilities from job post"""
        job_post = job_post.lower()
        sections = {
            'requirements': '',
            'responsibilities': '',
            'skills_needed': '',
            'full_text': job_post
        }
        
        req_keywords = ['requirements', 'required', 'must have', 'qualifications', 'prerequisites']
        resp_keywords = ['responsibilities', 'duties', 'role', 'you will', 'responsible for']
        skill_keywords = ['skills', 'technologies', 'tools', 'programming', 'software']
        
        paragraphs = [p.strip() for p in job_post.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            if any(keyword in paragraph_lower for keyword in req_keywords):
                sections['requirements'] += ' ' + paragraph
            elif any(keyword in paragraph_lower for keyword in resp_keywords):
                sections['responsibilities'] += ' ' + paragraph
            elif any(keyword in paragraph_lower for keyword in skill_keywords):
                sections['skills_needed'] += ' ' + paragraph
        
        return sections
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 0.0

        embeddings = self.model.encode([text1, text2])

        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def analyze_specific_matches(self, user_sections: Dict[str, str], job_sections: Dict[str, str]) -> Dict[str, Any]:
        """Analyze specific section matches"""
        matches = {}

        skills_similarity = self.calculate_similarity(
            user_sections['skills'] or user_sections['full_text'], 
            job_sections['skills_needed'] or job_sections['requirements']
        )
        matches['skills_match'] = {
            'score': skills_similarity,
            'level': self.get_match_level(skills_similarity)
        }

        exp_similarity = self.calculate_similarity(
            user_sections['experience'] or user_sections['full_text'], 
            job_sections['responsibilities'] or job_sections['full_text']
        )
        matches['experience_match'] = {
            'score': exp_similarity,
            'level': self.get_match_level(exp_similarity)
        }

        overall_similarity = self.calculate_similarity(
            user_sections['full_text'], 
            job_sections['requirements'] or job_sections['full_text']
        )
        matches['overall_match'] = {
            'score': overall_similarity,
            'level': self.get_match_level(overall_similarity)
        }
        
        return matches
    
    def get_match_level(self, score: float) -> str:
        """Convert similarity score to human-readable match level"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Very Good"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.35:
            return "Fair"
        else:
            return "Poor"
    
    def generate_recommendations(self, matches: Dict[str, Any], user_sections: Dict[str, str], job_sections: Dict[str, str]) -> List[str]:
        """Generate recommendations for improving job fit"""
        recommendations = []
        
        if matches['skills_match']['score'] < 0.5:
            recommendations.append("Consider highlighting more relevant technical skills that match the job requirements.")
        
        if matches['experience_match']['score'] < 0.5:
            recommendations.append("Emphasize experiences that align more closely with the job responsibilities.")
        
        if matches['overall_match']['score'] < 0.6:
            recommendations.append("Tailor your CV and cover letter to better match the specific job requirements.")
            recommendations.append("Use keywords from the job posting in your application materials.")
        
        if not user_sections['skills'].strip():
            recommendations.append("Add a dedicated skills section to your CV to better showcase your technical abilities.")
        
        if len(recommendations) == 0:
            recommendations.append("Your profile shows a strong match for this position!")
        
        return recommendations
    
    async def analyze_fit(self, user_profile: str, job_post: str) -> Dict[str, Any]:
        """Main method to analyze job fit"""
        try:
            user_sections = self.extract_key_sections(user_profile)
            job_sections = self.extract_job_requirements(job_post)

            matches = self.analyze_specific_matches(user_sections, job_sections)

            overall_score = (
                matches['skills_match']['score'] * 0.4 +
                matches['experience_match']['score'] * 0.3 +
                matches['overall_match']['score'] * 0.3
            )

            recommendations = self.generate_recommendations(matches, user_sections, job_sections)
            
            return {
                'overall_fit_score': round(overall_score, 3),
                'overall_fit_level': self.get_match_level(overall_score),
                'detailed_matches': matches,
                'recommendations': recommendations,
                'analysis_summary': f"Your profile shows a {self.get_match_level(overall_score).lower()} match for this position with an overall fit score of {round(overall_score * 100, 1)}%."
            }
        
        except Exception as e:
            return {
                'error': f"Analysis failed: {str(e)}",
                'overall_fit_score': 0.0,
                'overall_fit_level': 'Error',
                'detailed_matches': {},
                'recommendations': ['Please try again or check your input text.'],
                'analysis_summary': 'Analysis could not be completed due to an error.'
            }