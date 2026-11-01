@echo off
title Iris Botanical Analytics Lab - Streamlit Dashboard
echo =====================================================================
echo  Iris Botanical Analytics Lab - Multidimensional EDA Dashboard
echo =====================================================================
echo.
echo Installing requirements if needed...
python -m pip install -r requirements-streamlit.txt --quiet
echo.
echo Launching Streamlit Dashboard on http://localhost:8501 ...
python -m streamlit run streamlit_app.py
pause
