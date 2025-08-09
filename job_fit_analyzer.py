from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Optional, Tuple, Set
import os
import re
import numpy as np

class JobFitAnalyzer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        default_section_weights: Optional[Dict[str, float]] = None,
        predefined_skills: Optional[Set[str]] = None,
    ):
        """Initialize the job fit analyzer with pre-trained models and configuration.

        - model_name: Override embedding model; falls back to JOBFIT_MODEL env or MiniLM.
        - thresholds: Dict with keys: excellent, very_good, good, fair.
        - default_section_weights: Dict with keys: skills, experience, overall.
        - predefined_skills: Override default skills vocabulary.
        """
        configured_model = (
            model_name
            or os.environ.get("JOBFIT_MODEL")
            or "all-MiniLM-L6-v2"
        )
        print(f"Loading sentence transformer model: {configured_model}...")
        self.model = SentenceTransformer(configured_model)
        print("Model loaded successfully!")

        self.thresholds = thresholds or {
            "excellent": 0.80,
            "very_good": 0.65,
            "good": 0.50,
            "fair": 0.35,
        }

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return default

        self.thresholds["excellent"] = _env_float("JOBFIT_THRESHOLD_EXCELLENT", self.thresholds["excellent"])
        self.thresholds["very_good"] = _env_float("JOBFIT_THRESHOLD_VERY_GOOD", self.thresholds["very_good"])
        self.thresholds["good"] = _env_float("JOBFIT_THRESHOLD_GOOD", self.thresholds["good"])
        self.thresholds["fair"] = _env_float("JOBFIT_THRESHOLD_FAIR", self.thresholds["fair"])

        self.default_section_weights = default_section_weights or {
            "skills": 0.40,
            "experience": 0.30,
            "overall": 0.30,
        }

        base_skills = {
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", "r",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision",
            "sql", "postgresql", "mysql", "nosql", "mongodb", "data analysis", "data engineering",
            "docker", "kubernetes", "terraform", "aws", "azure", "gcp", "ci/cd", "linux",
            "react", "vue", "angular", "node.js", "django", "flask", "fastapi", "spring", ".net",
            "git", "agile", "scrum", "jira", "rest", "graphql", "microservices",
            "project management", "communication", "leadership", "problem solving", "testing",
        }
        self.predefined_skills: Set[str] = {s.lower() for s in (predefined_skills or base_skills)}

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing boilerplate, collapsing whitespace, and separating sentences.
        Keeps case to preserve entities, but trims noise.
        """
        if not text:
            return ""

        boilerplate_patterns = [
            r"^\s*dear (hiring manager|sir|madam).*?$",
            r"thank you for your consideration\.?",
            r"i look forward to (hearing from you|your response).*?",
            r"references available upon request\.?",
        ]

        cleaned = text
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        cleaned = re.sub(r"([.!?])\s+", r"\1\n", cleaned)
        return cleaned.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not parts:
            parts = [p.strip() for p in text.split("\n") if p.strip()]
        return parts

    def _embed_paragraphs(self, paragraphs: List[str]) -> np.ndarray:
        if not paragraphs:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=float)
        return np.asarray(self.model.encode(paragraphs))

    def _classify_paragraphs_semantically(self, paragraphs: List[str]) -> List[str]:
        """Classify each paragraph as one of: skills, experience, education, requirements, responsibilities, other.
        Uses semantic similarity to simple prototype descriptions.
        """
        if not paragraphs:
            return []

        labels = [
            ("skills", "This text lists specific skills, tools, technologies, or competencies."),
            ("experience", "This text describes prior work experience, roles, achievements, projects."),
            ("education", "This text lists education, degrees, certifications, courses, or universities."),
            ("requirements", "This text lists job requirements or qualifications for the position."),
            ("responsibilities", "This text describes responsibilities, duties, and what you will do in the role."),
        ]
        label_texts = [desc for _, desc in labels]
        label_names = [name for name, _ in labels]

        label_embeddings = self._embed_paragraphs(label_texts)
        para_embeddings = self._embed_paragraphs(paragraphs)

        assignments: List[str] = []
        if para_embeddings.shape[0] == 0:
            return ["other"] * len(paragraphs)

        similarity_matrix = cosine_similarity(para_embeddings, label_embeddings)
        for row in similarity_matrix:
            best_idx = int(np.argmax(row))
            assignments.append(label_names[best_idx])
        return assignments

    def _extract_skills_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()
        lowered = text.lower()
        found = {skill for skill in self.predefined_skills if skill in lowered}
        return found
    
    def extract_key_sections(self, text: str) -> Dict[str, Any]:
        """Extract key sections from CV/cover letter text using normalization, semantic cues, and skill extraction."""
        normalized = self.normalize_text(text)
        paragraphs = self._split_paragraphs(normalized)

        skill_keywords = ["skills", "technical skills", "competencies", "abilities", "technologies"]
        exp_keywords = ["experience", "work experience", "employment", "career", "professional"]
        edu_keywords = ["education", "qualification", "degree", "university", "college", "certification"]

        sections: Dict[str, str] = {
            "skills": "",
            "experience": "",
            "education": "",
            "full_text": normalized,
        }

        semantic_labels = self._classify_paragraphs_semantically(paragraphs)

        for paragraph, sem_label in zip(paragraphs, semantic_labels):
            lower = paragraph.lower()
            if any(k in lower for k in skill_keywords) or sem_label == "skills":
                sections["skills"] += " " + paragraph
            elif any(k in lower for k in exp_keywords) or sem_label == "experience":
                sections["experience"] += " " + paragraph
            elif any(k in lower for k in edu_keywords) or sem_label == "education":
                sections["education"] += " " + paragraph

        extracted_skills = self._extract_skills_from_text(normalized)
        return {**sections, "extracted_skills": sorted(extracted_skills)}
    
    def extract_job_requirements(self, job_post: str) -> Dict[str, Any]:
        """Extract requirements, responsibilities, and skills from job post using semantic classification and skill extraction."""
        normalized = self.normalize_text(job_post)
        paragraphs = self._split_paragraphs(normalized)

        sections: Dict[str, str] = {
            "requirements": "",
            "responsibilities": "",
            "skills_needed": "",
            "full_text": normalized,
        }

        semantic_labels = self._classify_paragraphs_semantically(paragraphs)
        for paragraph, sem_label in zip(paragraphs, semantic_labels):
            if sem_label == "requirements":
                sections["requirements"] += " " + paragraph
            elif sem_label == "responsibilities":
                sections["responsibilities"] += " " + paragraph
            elif sem_label == "skills":
                sections["skills_needed"] += " " + paragraph

        extracted_skills = self._extract_skills_from_text(normalized)
        return {**sections, "extracted_skills": sorted(extracted_skills)}
    
    def _chunk_text(self, text: str, max_words_per_chunk: int = 220) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        chunks: List[str] = []
        for i in range(0, len(words), max_words_per_chunk):
            chunk = " ".join(words[i : i + max_words_per_chunk])
            chunks.append(chunk)
        return chunks

    def _embed_text_avg(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros((self.model.get_sentence_embedding_dimension(),), dtype=float)
        chunks = self._chunk_text(text)
        vectors = np.asarray(self.model.encode(chunks))
        return vectors.mean(axis=0)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts with chunking to avoid truncation."""
        if not text1 or not text2 or not text1.strip() or not text2.strip():
            return 0.0

        vec1 = self._embed_text_avg(text1)
        vec2 = self._embed_text_avg(text2)
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return float(similarity)
    
    def analyze_specific_matches(self, user_sections: Dict[str, Any], job_sections: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific section matches and identify overlapping/missing skills."""
        matches: Dict[str, Any] = {}

        skills_similarity = self.calculate_similarity(
            user_sections.get("skills") or user_sections.get("full_text"),
            job_sections.get("skills_needed") or job_sections.get("requirements"),
        )
        matches["skills_match"] = {
            "score": skills_similarity,
            "level": self.get_match_level(skills_similarity),
        }

        exp_similarity = self.calculate_similarity(
            user_sections.get("experience") or user_sections.get("full_text"),
            job_sections.get("responsibilities") or job_sections.get("full_text"),
        )
        matches["experience_match"] = {
            "score": exp_similarity,
            "level": self.get_match_level(exp_similarity),
        }

        overall_similarity = self.calculate_similarity(
            user_sections.get("full_text"),
            job_sections.get("requirements") or job_sections.get("full_text"),
        )
        matches["overall_match"] = {
            "score": overall_similarity,
            "level": self.get_match_level(overall_similarity),
        }

        user_skill_set = set(user_sections.get("extracted_skills", []))
        job_skill_set = set(job_sections.get("extracted_skills", []))
        overlapping_skills = sorted(user_skill_set & job_skill_set)
        missing_skills = sorted(job_skill_set - user_skill_set)
        matches["overlapping_skills"] = overlapping_skills
        matches["missing_skills"] = missing_skills

        return matches
    
    def get_match_level(self, score: float) -> str:
        """Convert similarity score to human-readable match level using configured thresholds."""
        t = self.thresholds
        if score >= t["excellent"]:
            return "Excellent"
        if score >= t["very_good"]:
            return "Very Good"
        if score >= t["good"]:
            return "Good"
        if score >= t["fair"]:
            return "Fair"
        return "Poor"
    
    def generate_recommendations(self, matches: Dict[str, Any], user_sections: Dict[str, Any], job_sections: Dict[str, Any]) -> List[str]:
        """Generate concrete recommendations using missing skills and weak section matches."""
        recommendations: List[str] = []

        missing_skills = matches.get("missing_skills", [])
        if missing_skills:
            top_missing = ", ".join(missing_skills[:8])
            recommendations.append(
                f"Consider upskilling or highlighting exposure to: {top_missing}."
            )

        if matches["experience_match"]["score"] < max(self.thresholds["good"], 0.5):
            recommendations.append(
                "Strengthen your experience section with concrete achievements that match the listed responsibilities."
            )

        if matches["skills_match"]["score"] < max(self.thresholds["good"], 0.5):
            recommendations.append(
                "Ensure the most relevant tools and technologies for this role are prominent in your skills section."
            )

        if matches["overall_match"]["score"] < max(self.thresholds["very_good"], 0.6):
            recommendations.append(
                "Tailor the CV and cover letter wording to mirror phrasing from the job requirements for better ATS alignment."
            )

        if not (user_sections.get("skills") or "").strip():
            recommendations.append(
                "Add a dedicated skills section to your CV to better showcase your technical abilities."
            )

        if len(recommendations) == 0:
            recommendations.append("Your profile shows a strong match for this position!")

        return recommendations
    
    async def analyze_fit(
        self,
        user_profile: str,
        job_post: str,
        section_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Main method to analyze job fit.
        section_weights: optional override for weights with keys: skills, experience, overall
        """
        try:
            user_sections = self.extract_key_sections(user_profile)
            job_sections = self.extract_job_requirements(job_post)

            matches = self.analyze_specific_matches(user_sections, job_sections)
            overlapping_skills = matches.pop("overlapping_skills", [])
            missing_skills = matches.pop("missing_skills", [])

            weights = section_weights or self.default_section_weights
            skills_w = float(weights.get("skills", 0.4))
            exp_w = float(weights.get("experience", 0.3))
            overall_w = float(weights.get("overall", 0.3))
            total = skills_w + exp_w + overall_w
            if total <= 0:
                skills_w, exp_w, overall_w = 0.4, 0.3, 0.3
                total = 1.0
            skills_w /= total
            exp_w /= total
            overall_w /= total

            overall_score = (
                matches["skills_match"]["score"] * skills_w
                + matches["experience_match"]["score"] * exp_w
                + matches["overall_match"]["score"] * overall_w
            )

            recommendations = self.generate_recommendations(matches, user_sections, job_sections)

            return {
                "overall_fit_score": round(overall_score, 3),
                "overall_fit_level": self.get_match_level(overall_score),
                "detailed_matches": matches,
                "overlapping_skills": overlapping_skills,
                "missing_skills": missing_skills,
                "recommendations": recommendations,
                "analysis_summary": (
                    f"Your profile shows a {self.get_match_level(overall_score).lower()} match for this position "
                    f"with an overall fit score of {round(overall_score * 100, 1)}%."
                ),
            }

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "overall_fit_score": 0.0,
                "overall_fit_level": "Error",
                "detailed_matches": {},
                "recommendations": ["Please try again or check your input text."],
                "analysis_summary": "Analysis could not be completed due to an error.",
            }