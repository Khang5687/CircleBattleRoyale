# CUDA Battle Royale Simulation

A high-performance 2D battle royale simulation capable of handling up to 1 million circles with real-time physics, collision detection, and image-based rendering. Optimized for NVIDIA RTX 3060 laptop GPU.

## Features

- Real-time physics simulation for up to 1M circles
- CUDA-accelerated collision detection with spatial hashing
- Dynamic LOD-based rendering with texture atlases
- Configurable bias system for win probability
- Dynamic damage scaling based on population
- Performance metrics overlay

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 8.6 (RTX 3060 or equivalent)
- 6 GB VRAM minimum
- 8 GB System RAM minimum

### Software
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler
  - Windows: Visual Studio 2019 or later
  - Linux: GCC 9.0+ or Clang 10.0+
- OpenGL 4.6 support

## Building

### Windows

1. Install CUDA Toolkit from NVIDIA website
2. Install Visual Studio 2019 or later with C++ development tools
3. Clone the repository with submodules:
   ```
   git clone --recursive https://github.com/yourusername/cuda-battle-royale.git
   cd cuda-battle-royale
   ```

4. Create build directory and configure:
   ```
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```

5. Build the project:
   ```
   cmake --build . --config Release
   ```

6. Run the executable:
   ```
   bin\Release\CudaBattleRoyale.exe
   ```

### Linux

1. Install CUDA Toolkit:
   ```
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. Install dependencies:
   ```
   sudo apt-get install cmake build-essential libgl1-mesa-dev libglu1-mesa-dev
   ```

3. Clone the repository with submodules:
   ```
   git clone --recursive https://github.com/yourusername/cuda-battle-royale.git
   cd cuda-battle-royale
   ```

4. Create build directory and configure:
   ```
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

5. Build the project:
   ```
   make -j$(nproc)
   ```

6. Run the executable:
   ```
   ./bin/CudaBattleRoyale
   ```

## Dependencies

The project uses the following external libraries:

- **GLFW**: Window and input management (included as submodule)
- **Dear ImGui**: UI overlay (included as submodule)
- **stb_image**: Image loading (header-only, included)
- **CUDA**: GPU acceleration
- **OpenGL**: Rendering

## Configuration

Configuration files are located in the `config/` directory:

- `config.txt`: Main simulation parameters (circle count, arena size, physics)
- `bias.txt`: Player bias multipliers for win probability
- `damage_scaling.txt`: Dynamic damage scaling based on population

Avatar images should be placed in `assets/players/` directory as JPG or PNG files.

## Controls

- **F3**: Toggle performance metrics overlay
- **+/-**: Zoom in/out
- **Mouse**: Pan camera (when implemented)
- **ESC**: Exit simulation

## Project Structure

```
cuda-battle-royale/
├── src/                    # Source files
├── include/                # Header files
├── shaders/                # GLSL shader files
├── assets/                 # Avatar images
│   └── players/           # Player avatar directory
├── config/                 # Configuration files
├── external/              # External dependencies
│   ├── glfw/             # GLFW library
│   ├── imgui/            # Dear ImGui
│   └── stb/              # stb_image
├── build/                 # Build directory (generated)
└── CMakeLists.txt        # CMake configuration
```

## Performance Targets

- 60+ FPS with up to 100,000 circles
- 30+ FPS with 500,000 circles
- 15+ FPS with 1,000,000 circles

## License

[Your License Here]

## Acknowledgments

- NVIDIA CUDA Programming Guide
- GPU Gems 3 - N-body Simulation
- Spatial Hashing for Collision Detection
