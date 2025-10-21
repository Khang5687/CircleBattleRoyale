#!/bin/bash

echo "Building CUDA Battle Royale Simulation..."
echo ""

# Check if dependencies are set up
if [ ! -d "external/glfw" ]; then
    echo "Error: Dependencies not found!"
    echo "Please run ./setup_dependencies.sh first."
    exit 1
fi

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build
echo ""
echo "Building project..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

cd ..

echo ""
echo "Build complete! Executable is in build/bin/"
echo ""
