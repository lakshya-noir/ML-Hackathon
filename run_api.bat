@echo off
REM Laptop 1: start the PathSnap API (all interfaces, port 8000). Run from anywhere.
REM   run_api.bat                       -> serves data\student_test
REM   run_api.bat data\evaluator_test   -> serves the evaluator trips
cd /d "%~dp0"
if "%~1"=="" (python -m pathsnap.api.app --host 0.0.0.0 --port 8000) else (python -m pathsnap.api.app --host 0.0.0.0 --port 8000 --trips-dir "%~1")
