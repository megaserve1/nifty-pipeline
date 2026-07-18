@echo off
setlocal EnableExtensions
title NIFTY Pipeline - Windows and ClearML Setup

rem -----------------------------------------------------------------------------
rem Change these values only if the repository, install folder, or queue changes.
rem This file never stores GitHub, ClearML, or Google Cloud credentials.
rem -----------------------------------------------------------------------------
set "REPO_URL=https://github.com/megaserve1/nifty-pipeline.git"
set "INSTALL_DIR=%USERPROFILE%\nifty-pipeline"
set "PYTHON_VERSION=3.12"
set "CLEARML_QUEUE=training"

echo.
echo ================================================================
echo   NIFTY PIPELINE - WINDOWS SETUP
echo ================================================================
echo Repository : %REPO_URL%
echo Install at : %INSTALL_DIR%
echo.

rem ---- Required programs -------------------------------------------------------
where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git is not installed or is not on PATH.
    echo Install Git for Windows, reopen Command Prompt, and run this file again.
    goto :failed
)

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: The Windows Python launcher "py" was not found.
    echo Install 64-bit Python %PYTHON_VERSION%, including the Python launcher.
    goto :failed
)

py -%PYTHON_VERSION% --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python %PYTHON_VERSION% is not installed.
    echo Install 64-bit Python %PYTHON_VERSION% and run this file again.
    goto :failed
)

rem ---- Clone -------------------------------------------------------------------
if exist "%INSTALL_DIR%\.git" (
    echo [1/7] Repository already exists. Keeping it unchanged.
) else (
    if exist "%INSTALL_DIR%" (
        echo ERROR: %INSTALL_DIR% exists but is not a Git repository.
        echo Rename or remove that folder, then run this file again.
        goto :failed
    )

    echo [1/7] Cloning the private GitHub repository...
    git clone "%REPO_URL%" "%INSTALL_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: Git clone failed.
        echo If GitHub says "Repository not found", the account must be added as a
        echo collaborator, accept the invitation, and authenticate with GitHub first.
        goto :failed
    )
)

cd /d "%INSTALL_DIR%"
if errorlevel 1 goto :failed

rem ---- Project environment ------------------------------------------------------
echo [2/7] Creating the project virtual environment...
if not exist "final_venv\Scripts\python.exe" (
    py -%PYTHON_VERSION% -m venv final_venv
    if errorlevel 1 goto :failed
) else (
    echo       final_venv already exists.
)

set "VENV_PY=%INSTALL_DIR%\final_venv\Scripts\python.exe"

echo [3/7] Installing the pipeline dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

rem ---- Agent is intentionally outside final_venv -------------------------------
echo [4/7] Installing ClearML Agent for the Windows user...
py -%PYTHON_VERSION% -m pip install --user --upgrade clearml-agent
if errorlevel 1 goto :failed

for /f "delims=" %%I in ('py -%PYTHON_VERSION% -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"') do set "PY_USER_SCRIPTS=%%I"
set "CLEARML_AGENT_EXE=%PY_USER_SCRIPTS%\clearml-agent.exe"
if not exist "%CLEARML_AGENT_EXE%" (
    echo ERROR: clearml-agent installed, but its executable was not found at:
    echo        %CLEARML_AGENT_EXE%
    goto :failed
)

rem ---- ClearML credentials ------------------------------------------------------
echo [5/7] Checking ClearML configuration...
if exist "%USERPROFILE%\clearml.conf" (
    echo       Found %USERPROFILE%\clearml.conf
) else (
    echo.
    echo ClearML credentials are required now.
    echo In ClearML: Settings - Workspace - Create new credentials - Copy.
    echo Paste the copied block into the setup wizard.
    echo For the private GitHub repository, also configure Git authentication when asked.
    echo.
    "%CLEARML_AGENT_EXE%" init
    if errorlevel 1 (
        echo ERROR: ClearML Agent configuration was not completed.
        goto :failed
    )
)

echo.
echo IMPORTANT: compare this machine's ClearML configuration with:
echo   %INSTALL_DIR%\clearml.conf.example
echo Confirm the GCS project and output bucket before publishing real data.
echo.

rem ---- Quick offline verification ----------------------------------------------
echo [6/7] Running the small offline end-to-end test...
"%VENV_PY%" -m pytest -q tests\test_end_to_end_mini.py::test_the_whole_chain_register_freeze_build_certify_load
if errorlevel 1 (
    echo ERROR: Installation finished, but the pipeline smoke test failed.
    goto :failed
)

rem ---- Register the five reusable ClearML tasks --------------------------------
echo [7/7] ClearML base-task registration.
choice /C YN /N /M "Register the five ClearML base tasks now? [Y/N]: "
if errorlevel 2 goto :skip_registration

"%VENV_PY%" trainer\register_base_trainer.py
if errorlevel 1 (
    echo ERROR: Base-task registration failed. Check clearml.conf and network access.
    goto :failed
)

:skip_registration
echo.
echo ================================================================
echo   SETUP COMPLETE
echo ================================================================
echo Project folder : %INSTALL_DIR%
echo Project Python : %VENV_PY%
echo.
echo Before using DVC or GCS on a laptop, configure Google Cloud ADC.
echo To process ClearML jobs, start a worker in a NEW Command Prompt:
echo.
echo   "%CLEARML_AGENT_EXE%" daemon --queue %CLEARML_QUEUE%
echo.
echo The full executable path is shown so this works even when the Python user
echo Scripts folder is not on PATH.
echo.
pause
exit /b 0

:failed
echo.
echo Setup stopped because a step failed. Nothing was pushed to GitHub, GCS, or ClearML.
pause
exit /b 1
