@echo off
echo ------------------------------------------------------
echo   RESTARTING LIGHTRAG WITH ENTITY QUERY FIX
echo ------------------------------------------------------
echo.
echo This script will:
echo 1. Stop any running LightRAG server
echo 2. Clear all caches to apply the entity query fix
echo 3. Start the LightRAG server with the new entity query support
echo.

REM Stop any running server
echo Stopping any running LightRAG server instances...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *lightrag*" > nul 2>&1

REM Clear all caches
echo.
echo Clearing all caches...
python clear_cache.py

REM Start the server
echo.
echo Starting LightRAG server with entity query support...
echo This may take a moment...
start "LightRAG Server" /B python -m lightrag.api.lightrag_server

echo.
echo LightRAG server restarted with entity query support!
echo You can now query about specific entities like "Who is Paul Bessems?"
echo.
echo The server is running in background. Access the web interface at:
echo http://localhost:9621
echo.
pause 