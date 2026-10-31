@echo off
setlocal
title Install Resume Ranker (one time)
cd /d "%~dp0"
if not exist "app\app.py" ( echo ERROR: put this file in the bias_detection_project folder next to app, code and artifacts. & pause & exit /b 1 )
set PYCMD=
where py >nul 2>nul && set PYCMD=py
if "%PYCMD%"=="" ( where python >nul 2>nul && set PYCMD=python )
if "%PYCMD%"=="" ( echo ERROR: Install Python 3.11 or 3.12 from python.org first, tick "Add Python to PATH". & pause & exit /b 1 )
if not exist "venv\Scripts\python.exe" ( %PYCMD% -m venv venv || ( echo Could not create environment. & pause & exit /b 1 ) )
echo Installing, this takes 5-10 minutes, one time only...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || ( echo torch install failed. & pause & exit /b 1 )
venv\Scripts\python.exe -m pip install -r app\requirements.txt || ( echo Package install failed. & pause & exit /b 1 )
powershell -NoProfile -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Resume Ranker.lnk'); $s.TargetPath='%CD%\Resume Ranker.vbs'; $s.WorkingDirectory='%CD%'; $s.IconLocation='shell32.dll,44'; $s.Save()"
echo.
echo DONE. Open "Resume Ranker" from your Desktop.
pause
