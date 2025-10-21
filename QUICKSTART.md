# Quick Start Guide

## Prerequisites

1. **NVIDIA GPU** with Compute Capability 8.6 (RTX 3060 or equivalent)
2. **CUDA Toolkit** 11.0 or later installed
3. **CMake** 3.18 or later
4. **C++ Compiler** (Visual Studio 2019+ on Windows, GCC 9+ on Linux)

## Setup Steps

### 1. Download Dependencies

Run the setup script to download external libraries:

**Windows:**
```
setup_dependencies.bat
```

**Linux:**
```
chmod +x setup_dependencies.sh
./setup_dependencies.sh
```

This will download:
- GLFW (window management)
- Dear ImGui (UI overlay)
- stb_image (image loading)

### 2. Add Avatar Images

Place your avatar images in `assets/players/`:
- Supported formats: JPG, PNG
- Recommended size: 256x256 pixels
- Name them sequentially (e.g., `player_001.jpg`, `player_002.jpg`, etc.)

### 3. Configure Simulation

Edit configuration files in `config/`:

**config.txt** - Main simulation parameters:
```
initialCircleCount=10000
arenaWidth=10000.0
arenaHeight=10000.0
baseCircleRadius=50.0
baseCircleMass=1.0
elasticity=0.95
baseHealth=100.0
```

**bias.txt** - Player win probability bias (optional):
```
player_001 2.0
player_042 1.5
```

**damage_scaling.txt** - Dynamic damage scaling (optional):
```
100000 1.0
50000 1.5
10000 2.0
1000 3.0
```

### 4. Build the Project

**Windows:**
```
build.bat
```

**Linux:**
```
chmod +x build.sh
./build.sh
```

### 5. Run the Simulation

**Windows:**
```
build\bin\Release\CudaBattleRoyale.exe
```

**Linux:**
```
./build/bin/CudaBattleRoyale
```

## Controls

- **F3**: Toggle performance metrics
- **+/-**: Zoom in/out
- **ESC**: Exit

## Troubleshooting

### "CUDA not found"
- Ensure CUDA Toolkit is installed
- Add CUDA bin directory to PATH
- Restart terminal/IDE after installation

### "Failed to initialize GLFW"
- Update graphics drivers
- Ensure OpenGL 4.6 support

### "Out of memory"
- Reduce `initialCircleCount` in config.txt
- Reduce avatar image sizes
- Close other GPU-intensive applications

### Build errors
- Verify CUDA Toolkit version (11.0+)
- Check CMake version (3.18+)
- Ensure compute capability matches your GPU

## Next Steps

Once the basic simulation is running, you can:
1. Experiment with different circle counts
2. Add more avatar images
3. Adjust physics parameters
4. Configure bias for specific players
5. Tune damage scaling for faster/slower battles

For detailed documentation, see [README.md](README.md).
