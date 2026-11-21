@echo off
title AI Sound-to-Visual Translator - CNN Model Training
echo ========================================================
echo  ESC-50 CNN Model Training and Evaluation Pipeline
echo ========================================================
echo.
echo Starting standalone training on ESC-50 dataset...
python src/train_model.py
pause
