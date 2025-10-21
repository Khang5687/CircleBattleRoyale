# Technology Stack

## Core Technologies
- **CUDA**: GPU-accelerated physics and collision detection
- **OpenGL 4.6**: Rendering with instanced drawing
- **C++17**: Application logic and CPU-side code
- **CMake 3.18+**: Build system

## Libraries & Dependencies
- **GLFW**: Window and input management (submodule in `external/glfw`)
- **Dear ImGui**: UI overlay (submodule in `external/imgui`)
- **stb_image**: Image loading (header-only in `external/stb`)
- **CUDA Toolkit 11.0+**: Required for compilation

## Build System

### CMake Configuration
- Target architecture: CUDA Compute Capability 8.6 (RTX 3060)
- C++ standard: 17
- CUDA standard: 17
- CUDA flags: `--expt-relaxed-constexpr`, `-O3 -use_fast_math` (Release)

### Common Commands

**Windows:**
```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
bin\Release\CudaBattleRoyale.exe
```

**Linux:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./bin/CudaBattleRoyale
```

## Compiler Requirements
- **Windows**: Visual Studio 2019 or later with C++ development tools
- **Linux**: GCC 9.0+ or Clang 10.0+

## File Extensions
- `.cu`: CUDA source files (compiled with nvcc)
- `.cpp`: C++ source files
- `.h`: Header files
- `.vert/.frag`: GLSL shader files
