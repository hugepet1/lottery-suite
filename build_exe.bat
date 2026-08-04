@echo off
chcp 65001 >nul
set PYTHON=C:\Users\123\AppData\Local\Programs\Python\Python312\python.exe
cd /d "%~dp0"
"%PYTHON%" -m pip install -U pyinstaller openpyxl
"%PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "LotterySuite" ^
  --add-data "data\pl3;data/pl3" ^
  --add-data "data\dlt;data/dlt" ^
  --add-data "data\kl8;data/kl8" ^
  --hidden-import pl3.predict ^
  --hidden-import dlt.predict ^
  --hidden-import kl8.predict ^
  --hidden-import openpyxl ^
  app_gui.py
if not exist "dist\data\pl3" mkdir "dist\data\pl3"
if not exist "dist\data\dlt" mkdir "dist\data\dlt"
if not exist "dist\data\kl8" mkdir "dist\data\kl8"
xcopy /E /Y /I "data\pl3\*" "dist\data\pl3\"
xcopy /E /Y /I "data\dlt\*" "dist\data\dlt\"
xcopy /E /Y /I "data\kl8\*" "dist\data\kl8\"
copy /Y "dist\LotterySuite.exe" "dist\彩票号码助手.exe"
echo done
pause
