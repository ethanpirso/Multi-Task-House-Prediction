@echo off
echo Starting MLflow Tracking UI...
cd %~dp0
start "MLflow UI" py -m mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
echo Waiting for the server to start...
ping 127.0.0.1 -n 6 > nul
start http://localhost:8080
echo Navigate to http://localhost:8080 in your browser to view the results.
pause