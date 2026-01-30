@echo off
echo Starting PersonaPlex Desktop...
echo.
echo Step 1: Starting Python server in WSL2...
echo (This window will stay open - keep it running!)
echo.
wsl bash -c "cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server && source venv/bin/activate && export HF_TOKEN='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' && python main.py"
pause
