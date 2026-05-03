@echo off
cd /d C:\Users\Admin\Dev\div-transliteration
echo ================================================== >> train\training.log
echo Resumed at %date% %time% >> train\training.log
echo ================================================== >> train\training.log
venv\Scripts\python.exe train\finetune.py --resume >> train\training.log 2>&1
