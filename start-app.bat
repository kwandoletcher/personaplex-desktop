@echo off
echo Starting PersonaPlex Desktop Application...
echo.
echo Make sure the server is running first (start-server.bat)
echo.
cd /d C:\Users\User\Documents\moonshot\personaplex-desktop
npm run tauri-dev
pause
