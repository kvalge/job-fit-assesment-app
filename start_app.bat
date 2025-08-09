@echo off
echo Starting Job Fit Assessment App...
echo.
echo Installing dependencies if needed...
pip install -r requirements.txt
echo.
echo Starting server...
echo.
echo The app will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
uvicorn main:app --host 127.0.0.1 --port 8000 --reload