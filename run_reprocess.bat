@echo off
echo Running document reprocessing with entity extraction...
echo This may take some time due to rate limiting with Gemini's free tier.
echo.

python reprocess_entity_extraction.py

echo.
echo Process complete!
pause 