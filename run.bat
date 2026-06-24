@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Portfolio Manager...
streamlit run app.py
pause
