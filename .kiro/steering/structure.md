# Project Structure

## Directory Layout

```
cuda-battle-royale/
├── src/                    # Implementation files (.cpp, .cu)
├── include/                # Header files (.h)
├── shaders/                # GLSL shaders (.vert, .frag)
├── assets/players/         # Avatar images (JPG/PNG)
├── config/                 # Configuration files (.txt)
├── external/               # Third-party dependencies (submodules)
│   ├── glfw/
│   ├── imgui/
│   └── stb/
└── build/                  # Build output (generated, not in git)
```

## Component Organization

### Manager Pattern
Each major subsystem is implemented as a manager class with corresponding header and implementation files:

- **ConfigManager** (`config_manager.cpp/.h`): Configuration file parsing
- **TextureAtlasManager** (`texture_atlas_manager.cpp/.h`): Image loading and atlas creation
- **PhysicsManager** (`physics_manager.cu/.h`): CUDA physics simulation
- **CollisionManager** (`collision_manager.cu/.h`): CUDA collision detection
- **RenderingManager** (`rendering_manager.cpp/.h`): OpenGL rendering
- **UIManager** (`ui_manager.cpp/.h`): ImGui UI overlay

### File Naming Conventions
- Headers in `include/`, implementations in `src/`
- CUDA files use `.cu` extension
- C++ files use `.cpp` extension
- Headers use `.h` extension
- Manager classes use snake_case with `_manager` suffix

## Configuration Files

Located in `config/` directory:
- `config.txt`: Main simulation parameters (circle count, arena size, physics)
- `bias.txt`: Player bias multipliers for win probability
- `damage_scaling.txt`: Dynamic damage scaling based on population

## Asset Organization

- Avatar images: `assets/players/` (JPG or PNG format)
- Shaders: `shaders/` directory (vertex and fragment shaders)

## Build Artifacts

CMake generates build artifacts in `build/` directory:
- Executable: `build/bin/CudaBattleRoyale[.exe]`
- Assets, config, and shaders are copied to `build/bin/` during build
