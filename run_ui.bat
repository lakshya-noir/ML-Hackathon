@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:5500 and open it.
REM Then type Laptop 1 address, e.g. http://192.168.1.20:8000, into the API box.
cd /d "%~dp0ui"
start "" http://localhost:5500
python -m http.server 5500
