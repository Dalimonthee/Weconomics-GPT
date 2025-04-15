@echo off
echo ------------------------------------------------------
echo   IMPROVED ENTITY EXTRACTION WITH GENERALIZED TYPES
echo ------------------------------------------------------
echo.
echo This script will reprocess your document with improved
echo entity extraction using generalized entity types.
echo.
echo The process includes:
echo 1. Clearing the entity extraction cache
echo 2. Reprocessing with enhanced entity detection
echo 3. Using generalized entity types suitable for any domain
echo.
echo NOTE: This may take some time due to rate limiting with 
echo       the Gemini API free tier.
echo.

REM Stop any running LightRAG server first
echo Stopping any running LightRAG server instances...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *lightrag*" > nul 2>&1

REM Clear the cache and reprocess
echo.
echo Starting improved entity extraction...
python reprocess_entity_extraction.py

echo.
echo Process complete!
echo.
echo If you want to view the results, start the LightRAG server
echo with: python -m lightrag.api.lightrag_server
echo.
pause 