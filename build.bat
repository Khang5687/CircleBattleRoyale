@echo off
echo Building CUDA Battle Royale Simulation...
echo.

REM Check if dependencies are set up
if not exist external\glfw (
    echo Error: Dependencies not found!
    echo Please run setup_dependencies.bat first.
    pause
    exit /b 1
)

REM Create build directory
if not exist build (
    mkdir build
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 16 2019" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build
echo.
echo Building project...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

cd ..

echo.
echo Build complete! Executable is in build\bin\Release\
echo.
pause
