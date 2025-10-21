#!/bin/bash

echo "Setting up external dependencies for CUDA Battle Royale..."
echo ""

cd external

echo "Cloning GLFW..."
if [ ! -d "glfw" ]; then
    git clone https://github.com/glfw/glfw.git
    echo "GLFW cloned successfully"
else
    echo "GLFW already exists"
fi

echo ""
echo "Cloning Dear ImGui..."
if [ ! -d "imgui" ]; then
    git clone https://github.com/ocornut/imgui.git
    echo "Dear ImGui cloned successfully"
else
    echo "Dear ImGui already exists"
fi

echo ""
echo "Setting up stb_image..."
if [ ! -d "stb" ]; then
    mkdir stb
    cd stb
    curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    cd ..
    echo "stb_image downloaded successfully"
else
    echo "stb already exists"
fi

cd ..

echo ""
echo "Dependencies setup complete!"
echo "You can now build the project using CMake."
