@echo off

rem Preprocess data using OpenCV
cd ./OpenCV
cmake --build ./build
cd build/Debug
OpenCVExecutable.exe
cd ../../../

rem Set the path to mingw32-make and the required environment variables
set PATH=C:/raylib/w64devkit/bin;%PATH%
set RAYLIB_PATH=C:/raylib/raylib
set PROJECT_NAME=main
set OBJS=src/*.cpp
set BUILD_MODE=DEBUG

rem Execute the make command
mingw32-make.exe RAYLIB_PATH=%RAYLIB_PATH% PROJECT_NAME=%PROJECT_NAME% OBJS=%OBJS% BUILD_MODE=%BUILD_MODE%

rem Check if the build was successful
if errorlevel 1 (
    echo Build failed
    pause
    exit /b 1
)

rem Run the compiled application
%PROJECT_NAME%.exe

rem Pause after running
pause