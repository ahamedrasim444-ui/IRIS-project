@echo off
echo ========================================================
echo Push Iris EDA Dashboard to GitHub
echo ========================================================
echo.
echo Step 1: Default repository URL is set to:
echo         https://github.com/ahamedrasim444-ui/IRIS-mini.git
echo.
set "REPO_URL=https://github.com/ahamedrasim444-ui/IRIS-mini.git"

if "%REPO_URL%"=="" (
    echo Error: No repository URL provided.
    pause
    exit /b
)

echo.
echo Adding remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo Pushing branch 'main' to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo SUCCESS! Repository pushed to GitHub successfully!
    echo.
    echo Now open Vercel to deploy in 1 click:
    echo https://vercel.com/new
    echo ========================================================
) else (
    echo.
    echo Push failed. Please check your GitHub credentials or repository URL.
)

pause
