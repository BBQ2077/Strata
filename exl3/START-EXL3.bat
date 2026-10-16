@echo off
rem Strata (OpenAI / Anthropic API, web UI) on the EXL3 engine - native Windows, no WSL.
rem 1) the server's venv (.venv-exl3, Python 3.11 via uv)  2) the engine copied into the EXL3 runtime
rem 3) tokenizer/ written from the model folder (once)    4) the server on http://127.0.0.1:8080
rem Folders: STRATA_EXL3_RUNTIME (EXL3 runtime) and STRATA_EXL3_MODEL (EXL3 model), from the environment
rem or from exl3\local.bat (kept out of git; start from exl3\local.example.bat).
setlocal
cd /d "%~dp0.."
if exist "exl3\local.bat" call "exl3\local.bat"
if not defined STRATA_EXL3_RUNTIME goto :paths
if not defined STRATA_EXL3_MODEL goto :paths
set "RT=%STRATA_EXL3_RUNTIME%"
set "MODEL=%STRATA_EXL3_MODEL%"
for %%I in ("%MODEL%") do set "PACK=pack\exl3-%%~nxI"
set "STRATA_EXL3_TOKENIZER=%CD%\%PACK%\tokenizer"
set "PY=.venv-exl3\Scripts\python.exe"

if not exist "%PY%" (
  uv venv .venv-exl3 --python 3.11 || goto :fail
  uv pip install --python "%PY%" -r exl3\requirements-exl3.txt || goto :fail
)
copy /y exl3\strata_exl3_engine.py "%RT%\strata_exl3_engine.py" >nul || goto :fail
if not exist "%PACK%\tokenizer\vocab.json" (
  "%PY%" exl3\hf_tokenizer.py --model "%MODEL%" --out "%PACK%" --check || goto :fail
)
"%PY%" serve\server.py --engine strata --config exl3\strata-exl3.json --port 8080 %*
exit /b %errorlevel%

:paths
echo [strata-exl3] set STRATA_EXL3_RUNTIME and STRATA_EXL3_MODEL (copy exl3\local.example.bat to exl3\local.bat)
exit /b 1

:fail
echo [strata-exl3] setup step failed (see the message above)
exit /b 1
