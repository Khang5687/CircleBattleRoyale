# Architecture Overview

## Project Structure

```
cuda-battle-royale/
├── src/                          # Implementation files
│   ├── main.cpp                 # Entry point and main loop
│   ├── config_manager.cpp       # Configuration file parsing
│   ├── texture_atlas_manager.cpp # Image loading and atlas creation
│   ├── physics_manager.cu       # CUDA physics simulation
│   ├── collision_manager.cu     # CUDA collision detection
│   ├── rendering_manager.cpp    # OpenGL rendering
│   └── ui_manager.cpp           # ImGui UI overlay
│
├── include/                      # Header files
│   ├── config_manager.h
│   ├── texture_atlas_manager.h
│   ├── physics_manager.h
│   ├── collision_manager.h
│   ├── rendering_manager.h
│   └── ui_manager.h
│
├── shaders/                      # GLSL shaders
│   ├── circle.vert              # Vertex shader
│   └── circle.frag              # Fragment shader
│
├── config/                       # Configuration files
│   ├── config.txt               # Main simulation config
│   ├── bias.txt                 # Player bias settings
│   └── damage_scaling.txt       # Dynamic damage scaling
│
├── assets/                       # Runtime assets
│   └── players/                 # Avatar images (JPG/PNG)
│
└── external/                     # Third-party libraries
    ├── glfw/                    # Window management
    ├── imgui/                   # UI framework
    └── stb/                     # Image loading
```

## Component Architecture

### 1. Configuration Manager
- Loads and validates configuration files
- Provides access to simulation parameters
- Supports hot-reloading for bias and damage scaling

### 2. Texture Atlas Manager
- Loads avatar images from disk
- Packs images into texture atlases
- Generates mipmaps for LOD rendering
- Uploads to GPU with compression

### 3. Physics Manager
- Manages circle data (position, velocity, health)
- Updates physics using CUDA kernels
- Handles boundary collisions
- Provides device memory access

### 4. Collision Manager
- Implements spatial hashing for broad-phase
- Detects circle-circle collisions
- Applies elastic collision physics
- Calculates and applies damage

### 5. Rendering Manager
- Sets up OpenGL context and shaders
- Implements CUDA-OpenGL interop
- Performs instanced rendering
- Handles LOD selection

### 6. UI Manager
- Displays performance metrics
- Shows winner announcement
- Handles F3 toggle for visibility

## Data Flow

```
┌─────────────┐
│   Config    │
│   Files     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────┐
│         Initialization Phase            │
│  ┌────────────┐      ┌──────────────┐  │
│  │ Load Config│─────>│ Load Avatars │  │
│  └────────────┘      └──────┬───────┘  │
│                             │           │
│                             v           │
│                      ┌──────────────┐  │
│                      │Create Atlas  │  │
│                      └──────┬───────┘  │
│                             │           │
│                             v           │
│                      ┌──────────────┐  │
│                      │Alloc GPU Mem │  │
│                      └──────────────┘  │
└─────────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────┐
│          Simulation Loop (60 FPS)       │
│                                         │
│  ┌──────────────┐                      │
│  │Update Physics│ (CUDA)               │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │Build Spatial │ (CUDA)               │
│  │    Hash      │                      │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │   Detect     │ (CUDA)               │
│  │  Collisions  │                      │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │Apply Damage  │ (CUDA)               │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │Remove Dead   │ (CUDA)               │
│  │   Circles    │                      │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │   Render     │ (OpenGL)             │
│  │   Circles    │                      │
│  └──────┬───────┘                      │
│         │                               │
│         v                               │
│  ┌──────────────┐                      │
│  │  Update UI   │ (ImGui)              │
│  └──────────────┘                      │
│         │                               │
│         └──────> Check Winner          │
└─────────────────────────────────────────┘
```

## Memory Layout

### CPU Memory
- Configuration data (~1 KB)
- Atlas metadata (~100 KB for 1000 avatars)
- UI state (~10 KB)

### GPU Memory (6 GB VRAM)
- Circle data: 40 bytes × 1M = 40 MB
- Spatial hash grid: ~4 MB
- Texture atlases (compressed): ~85 MB per atlas
- OpenGL buffers: ~50 MB
- Total: ~200 MB for 1M circles + textures

## Performance Optimizations

### CUDA Optimizations
1. **Coalesced Memory Access**: Aligned data structures
2. **Shared Memory**: Cache frequently accessed data
3. **Occupancy**: 256 threads per block
4. **Streams**: Concurrent kernel execution

### Rendering Optimizations
1. **Instanced Rendering**: Single draw call
2. **LOD System**: Automatic mipmap selection
3. **Frustum Culling**: GPU-driven rendering
4. **Texture Compression**: BC1 format (75% savings)

### Memory Optimizations
1. **Zero-Copy**: CUDA-OpenGL interop
2. **Pinned Memory**: Faster CPU-GPU transfers
3. **Memory Pools**: Reusable buffers
4. **Compaction**: Free unused memory

## Build System

### CMake Configuration
- Minimum version: 3.18
- CUDA architecture: 8.6 (RTX 3060)
- C++ standard: 17
- CUDA standard: 17

### Compiler Flags
- **Release**: `-O3 -use_fast_math` (CUDA), `/O2` (MSVC)
- **Debug**: `-g -G` (CUDA), `/Zi` (MSVC)

### Dependencies
- CUDA Toolkit (required)
- OpenGL (required)
- GLFW (submodule)
- Dear ImGui (submodule)
- stb_image (header-only)

## Testing Strategy

### Unit Tests
- Configuration parsing
- Texture atlas creation
- Physics calculations
- Collision detection

### Integration Tests
- CUDA-OpenGL interop
- End-to-end simulation
- Memory leak detection

### Performance Tests
- FPS benchmarks at various scales
- Memory usage profiling
- Kernel performance analysis

## Future Enhancements

1. **Advanced Collision**: Quadtree or BVH
2. **Better Packing**: Bin-packing for texture atlas
3. **Compute Shaders**: Move more work to GPU
4. **Multi-GPU**: Scale to multiple GPUs
5. **Networking**: Distributed simulation
6. **Recording**: Save simulation replay
7. **Analytics**: Detailed statistics tracking
