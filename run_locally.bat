@echo off
title Iris EDA Streamlit Dashboard
echo =====================================================================
echo  Iris Dataset Multidimensional EDA Dashboard
echo  CSE Mini-Project
echo =====================================================================
echo.
echo Installing requirements if needed...
python -m pip install -r requirements.txt --quiet
echo.
echo Launching Streamlit Dashboard on http://localhost:8501 ...
python -m streamlit run app.py
pause
