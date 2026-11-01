@echo off
echo ========================================================
echo Starting Cloudflare Public HTTPS Tunnel for Iris EDA App
echo ========================================================
echo Routing traffic from public Internet to http://localhost:8501 using HTTP/2...
.\cloudflared.exe tunnel --url http://localhost:8501 --protocol http2
pause
