@echo off
REM Quick run script for CUDA Battle Royale Simulator
REM Assumes the project has already been built

cd build\bin\release
if exist CudaBattleRoyale.exe (
    echo Starting CUDA Battle Royale Simulator...
    CudaBattleRoyale.exe
) else (
    echo Error: CudaBattleRoyale.exe not found!
    echo Please build the project first using build.bat
    cd ..\..
    exit /b 1
)
cd ..\..
