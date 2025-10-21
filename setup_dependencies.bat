@echo off
echo Setting up external dependencies for CUDA Battle Royale...
echo.

cd external

echo Cloning GLFW...
if not exist glfw (
    git clone https://github.com/glfw/glfw.git
    echo GLFW cloned successfully
) else (
    echo GLFW already exists
)

echo.
echo Cloning Dear ImGui...
if not exist imgui (
    git clone https://github.com/ocornut/imgui.git
    echo Dear ImGui cloned successfully
) else (
    echo Dear ImGui already exists
)

echo.
echo Setting up stb_image...
if not exist stb (
    mkdir stb
    cd stb
    curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    cd ..
    echo stb_image downloaded successfully
) else (
    echo stb already exists
)

cd ..

echo.
echo Dependencies setup complete!
echo You can now build the project using CMake.
pause
