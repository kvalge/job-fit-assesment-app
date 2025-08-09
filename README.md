# Vibe Coding Project of Job Fit Assessment App

A FastAPI-based web application that uses Hugging Face models to analyze how well a user's CV and cover letter match a job posting.

## Features

- **CV & Cover Letter Storage**: Save your CV and cover letter text to files
- **Job Posting Analysis**: Analyze job fit by comparing your profile with job requirements
- **AI-Powered Matching**: Uses sentence transformers for semantic similarity analysis
- **Detailed Scoring**: Provides scores for skills match, experience match, and overall fit
- **Smart Recommendations**: Offers suggestions to improve your job application
- **Dual Clear Options**: 
  - Clear Fields (preserves saved data for reuse)
  - Clear All & Delete Saved (complete reset)

## Installation

**Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   The app uses these key dependencies:
   - FastAPI for the web framework
   - Sentence Transformers for AI text analysis
   - Scikit-learn for similarity calculations
   - Jinja2 for HTML templating

## Running the Application

1. **Start the FastAPI server**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the application**
   Open your browser and go to: `http://localhost:8000`

## How to Use

1. **CV Section**:
   - Choose "Enter new CV content" to input fresh CV text, or
   - Choose "Use saved CV" to use previously saved CV data
   - If entering new content, click "Save CV" to store it
   - If using saved data, click "Refresh" to update the preview

2. **Cover Letter Section**:
   - Choose "Enter new cover letter content" to input fresh text, or  
   - Choose "Use saved cover letter" to use previously saved data
   - If entering new content, click "Save Cover Letter" to store it
   - If using saved data, click "Refresh" to update the preview

3. **Analyze Job Fit**: 
   - Paste a job posting in the job post text area
   - Click "Analyze Job Fit" (this will auto-save any new CV/cover letter content)
   - Wait for the AI analysis to complete

4. **Review Results**:
   - Overall fit score (percentage and level)
   - Detailed breakdown of skills, experience, and overall matches
   - Personalized recommendations for improvement

5. **Clear Options**:
   - **Clear Fields**: Clears all input fields but keeps saved CV/cover letter files
   - **Clear All & Delete Saved**: Completely resets everything including deleting saved files

## How It Works

The application uses the `all-MiniLM-L6-v2` sentence transformer model from Hugging Face to:

1. **Extract Key Sections**: Automatically identifies skills, experience, and education sections from your CV and cover letter
2. **Analyze Job Requirements**: Parses the job posting to extract requirements, responsibilities, and needed skills
3. **Calculate Semantic Similarity**: Uses cosine similarity between text embeddings to measure how well your profile matches the job
4. **Generate Insights**: Provides detailed scoring and actionable recommendations

## File Structure

```
job-fit-assessment-app/
├── main.py                 # FastAPI application
├── job_fit_analyzer.py     # AI analysis logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── data/                  # Created automatically for storing CV/cover letter
│   ├── cv.json
│   └── cover_letter.json
└── README.md
```

## API Endpoints

- `GET /` - Main web interface
- `POST /save-cv` - Save CV text
- `POST /save-cover-letter` - Save cover letter text
- `POST /analyze-job-fit` - Analyze job fit
- `GET /get-saved-data` - Retrieve saved CV and cover letter

## Model Information

The app uses the `all-MiniLM-L6-v2` model which:
- Maps sentences to 384-dimensional dense vectors
- Is optimized for semantic similarity tasks
- Provides good performance with reasonable resource usage
- Downloads automatically on first run (~90MB)

## Tips for Best Results

1. **Complete Information**: Provide detailed CV and cover letter content
2. **Full Job Posting**: Include the complete job posting with requirements and responsibilities
3. **Relevant Keywords**: Use industry-specific terms that appear in the job posting
4. **Structure**: Organize your CV with clear sections for skills, experience, and education

## Troubleshooting

- **Model Loading**: First run may take longer as the AI model downloads
- **Memory Usage**: The app requires ~2GB RAM for optimal performance
- **Text Length**: Very long documents (>10,000 words) may take longer to process

## Future Enhancements

- Support for PDF file uploads
- Multiple job posting comparisons
- Historical analysis tracking
- More detailed skill gap analysis
- Integration with job boards