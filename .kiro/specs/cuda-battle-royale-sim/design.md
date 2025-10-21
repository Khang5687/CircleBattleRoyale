# Design Document

## Overview

This document describes the technical design for a high-performance 2D battle royale simulation system capable of handling up to 1 million circles with real-time physics, collision detection, and image-based rendering. The system leverages NVIDIA CUDA for parallel physics computation and OpenGL 4.6 for hardware-accelerated rendering, optimized specifically for the RTX 3060 laptop GPU (Ampere architecture, compute capability 8.6).

### Key Design Principles

1. **GPU-First Architecture**: All performance-critical operations execute on the GPU to maximize parallelism
2. **Zero-Copy Memory Sharing**: CUDA-OpenGL interop eliminates CPU-GPU data transfers
3. **Spatial Locality**: Data structures optimized for coalesced memory access patterns
4. **Scalable LOD System**: Automatic quality adjustment maintains 60+ FPS across all circle counts
5. **Modular Configuration**: Runtime-configurable parameters without recompilation

### Target Hardware Specifications

- **GPU**: NVIDIA RTX 3060 Laptop (Ampere GA106)
- **CUDA Cores**: 3,584
- **Compute Capability**: 8.6
- **Memory**: 6 GB GDDR6
- **Memory Bandwidth**: 360 GB/s
- **Tensor Cores**: 112 (3rd generation)
- **RT Cores**: 28 (2nd generation)

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Config Mgr   │  │  Input Mgr   │  │  UI Overlay  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Engine                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Physics Mgr  │  │ Collision Mgr│  │  Health Mgr  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      CUDA Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Physics Kernel│  │Spatial Hash  │  │ Damage Kernel│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Rendering Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Texture Atlas │  │  LOD Manager │  │OpenGL Renderer│     │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   GPU Hardware (RTX 3060)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ CUDA Cores   │  │Texture Units │  │  VRAM (12GB) │      │
│  │   (3,584)    │  │    (112)     │  │  360 GB/s    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Initialization Phase**:
   - Load configuration files (config.txt, bias.txt, damage_scaling.txt)
   - Load avatar images from assets/players directory
   - Create texture atlas with mipmaps
   - Allocate CUDA memory and OpenGL VBOs
   - Initialize spatial hashing grid

2. **Simulation Loop** (60+ FPS target):
   - **Physics Update** (CUDA): Update positions, velocities
   - **Collision Detection** (CUDA): Spatial hashing + narrow phase
   - **Damage Application** (CUDA): Apply collision damage with scaling
   - **Health Check** (CUDA): Remove dead circles, compact arrays
   - **Rendering** (OpenGL): Instanced rendering with LOD selection
   - **UI Update** (CPU): Performance metrics, winner announcement

3. **Termination**:
   - Declare winner when one circle remains
   - Display final statistics
   - Clean up GPU resources

## Components and Interfaces

### 1. Configuration Manager

**Responsibility**: Load and validate configuration files

**Interface**:
```cpp
class ConfigManager {
public:
    struct SimConfig {
        uint32_t initialCircleCount;
        float arenaWidth;
        float arenaHeight;
        float baseCircleRadius;
        float baseCircleMass;
        float elasticity;
        float baseHealth;
    };
    
    struct BiasEntry {
        std::string playerName;
        float biasMultiplier;
    };
    
    struct DamageScalingEntry {
        uint32_t populationThreshold;
        float damageMultiplier;
    };
    
    bool loadConfig(const std::string& configPath);
    bool loadBias(const std::string& biasPath);
    bool loadDamageScaling(const std::string& scalingPath);
    
    const SimConfig& getSimConfig() const;
    const std::vector<BiasEntry>& getBiasEntries() const;
    const std::vector<DamageScalingEntry>& getDamageScaling() const;
};
```

**Design Decisions**:
- Use simple text format for easy manual editing
- Validate all parameters on load to fail fast
- Support hot-reloading for bias and damage scaling files

### 2. Texture Atlas Manager

**Responsibility**: Load avatar images and create GPU texture atlas with mipmaps

**Interface**:
```cpp
class TextureAtlasManager {
public:
    struct AtlasEntry {
        uint32_t atlasIndex;
        float u0, v0, u1, v1;  // UV coordinates
        uint32_t width, height;
    };
    
    bool loadAvatars(const std::string& avatarDir);
    bool createAtlas(uint32_t maxAtlasSize = 8192);
    bool uploadToGPU();
    
    cudaTextureObject_t getCudaTextureObject() const;
    GLuint getOpenGLTextureID() const;
    const AtlasEntry& getAtlasEntry(uint32_t circleID) const;
    
private:
    std::vector<AtlasEntry> atlasEntries_;
    cudaArray_t cudaArray_;
    cudaTextureObject_t cudaTexture_;
    GLuint glTextureID_;
};
```

**Design Decisions**:
- Use stb_image for loading JPG/PNG files
- Pack images using simple row-based packing (can upgrade to bin-packing if needed)
- Generate 4 mipmap levels: full, 1/2, 1/4, 1/8 resolution
- Use BC1 compression for RGB channels to save 75% VRAM
- Maximum atlas size: 8192x8192 (supports ~1000 256x256 avatars per atlas)
- Create multiple atlases if needed for >1000 unique avatars

### 3. Physics Manager

**Responsibility**: Update circle positions and velocities using CUDA

**Interface**:
```cpp
class PhysicsManager {
public:
    struct Circle {
        float2 position;
        float2 velocity;
        float radius;
        float mass;
        float health;
        uint32_t avatarID;
        float biasMultiplier
;
    };
    
    bool initialize(uint32_t maxCircles, const ConfigManager::SimConfig& config);
    void updatePhysics(float deltaTime);
    void applyBoundaryConditions(float arenaWidth, float arenaHeight);
    
    Circle* getCircleData();
    uint32_t getActiveCircleCount() const;
    
private:
    Circle* d_circles_;  // Device memory
    uint32_t maxCircles_;
    uint32_t activeCount_;
    cudaStream_t physicsStream_;
};
```

**CUDA Kernel Design**:
```cpp
__global__ void updatePhysicsKernel(
    Circle* circles,
    uint32_t count,
    float deltaTime,
    float arenaWidth,
    float arenaHeight
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Circle& c = circles[idx];
    
    // Update position using Euler integration
    c.position.x += c.velocity.x * deltaTime;
    c.position.y += c.velocity.y * deltaTime;
    
    // Boundary collision (elastic bounce)
    if (c.position.x - c.radius < 0) {
        c.position.x = c.radius;
        c.velocity.x = -c.velocity.x;
    }
    if (c.position.x + c.radius > arenaWidth) {
        c.position.x = arenaWidth - c.radius;
        c.velocity.x = -c.velocity.x;
    }
    if (c.position.y - c.radius < 0) {
        c.position.y = c.radius;
        c.velocity.y = -c.velocity.y;
    }
    if (c.position.y + c.radius > arenaHeight) {
        c.position.y = arenaHeight - c.radius;
        c.velocity.y = -c.velocity.y;
    }
}
```

**Design Decisions**:
- Use simple Euler integration (sufficient for this use case)
- Each thread processes one circle (embarrassingly parallel)
- Block size: 256 threads (optimal for Ampere architecture)
- Use float2 for position/velocity (enables vectorized loads)

### 4. Collision Manager

**Responsibility**: Detect collisions using spatial hashing and apply physics

**Interface**:
```cpp
class CollisionManager {
public:
    bool initialize(uint32_t maxCircles, float arenaWidth, float arenaHeight, float cellSize);
    void detectAndResolveCollisions(PhysicsManager::Circle* circles, uint32_t count);
    
private:
    struct SpatialCell {
        uint32_t* circleIndices;
        uint32_t count;
    };
    
    SpatialCell* d_grid_;
    uint32_t gridWidth_;
    uint32_t gridHeight_;
    float cellSize_;
    cudaStream_t collisionStream_;
};
```

**Spatial Hashing Algorithm**:

1. **Grid Construction Phase**:
   - Divide arena into uniform grid cells
   - Cell size = 2 × max circle radius (ensures circles only span adjacent cells)
   - Each circle assigned to one primary cell based on center position

2. **Collision Detection Phase**:
   - For each circle, check collisions with circles in same cell and 8 adjacent cells
   - Use circle-circle intersection test: `distance < r1 + r2`

**CUDA Kernel Design**:
```cpp
__global__ void buildSpatialHashKernel(
    const Circle* circles,
    uint32_t count,
    SpatialCell* grid,
    uint32_t gridWidth,
    uint32_t gridHeight,
    float cellSize
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    const Circle& c = circles[idx];
    uint32_t cellX = (uint32_t)(c.position.x / cellSize);
    uint32_t cellY = (uint32_t)(c.position.y / cellSize);
    uint32_t cellIdx = cellY * gridWidth + cellX;
    
    // Atomic add to cell (thread-safe)
    uint32_t slot = atomicAdd(&grid[cellIdx].count, 1);
    grid[cellIdx].circleIndices[slot] = idx;
}

__global__ void detectCollisionsKernel(
    Circle* circles,
    uint32_t count,
    const SpatialCell* grid,
    uint32_t gridWidth,
    uint32_t gridHeight,
    float cellSize,
    float damageMultiplier
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Circle& c1 = circles[idx];
    uint32_t cellX = (uint32_t)(c1.position.x / cellSize);
    uint32_t cellY = (uint32_t)(c1.position.y / cellSize);
    
    // Check 3x3 neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cellX + dx;
            int ny = cellY + dy;
            if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight) continue;
            
            uint32_t cellIdx = ny * gridWidth + nx;
            const SpatialCell& cell = grid[cellIdx];
            
            for (uint32_t i = 0; i < cell.count; i++) {
                uint32_t otherIdx = cell.circleIndices[i];
                if (otherIdx <= idx) continue;  // Avoid duplicate checks
                
                Circle& c2 = circles[otherIdx];
                
                // Circle-circle collision test
                float dx = c2.position.x - c1.position.x;
                float dy = c2.position.y - c1.position.y;
                float distSq = dx*dx + dy*dy;
                float minDist = c1.radius + c2.radius;
                
                if (distSq < minDist * minDist && distSq > 0) {
                    // Collision detected - apply elastic collision physics
                    float dist = sqrtf(distSq);
                    float nx = dx / dist;
                    float ny = dy / dist;
                    
                    // Relative velocity
                    float dvx = c2.velocity.x - c1.velocity.x;
                    float dvy = c2.velocity.y - c1.velocity.y;
                    float dvn = dvx * nx + dvy * ny;
                    
                    if (dvn < 0) {  // Moving toward each other
                        // Elastic collision impulse
                        float impulse = 2 * dvn / (c1.mass + c2.mass);
                        
                        c1.velocity.x += impulse * c2.mass * nx;
                        c1.velocity.y += impulse * c2.mass * ny;
                        c2.velocity.x -= impulse * c1.mass * nx;
                        c2.velocity.y -= impulse * c1.mass * ny;
                        
                        // Apply damage based on collision force
                        float collisionForce = fabsf(dvn) * (c1.mass + c2.mass);
                        float baseDamage = collisionForce * 0.01f;  // Tunable constant
                        
                        atomicAdd(&c1.health, -baseDamage * damageMultiplier / c1.biasMultiplier);
                        atomicAdd(&c2.health, -baseDamage * damageMultiplier / c2.biasMultiplier);
                    }
                }
            }
        }
    }
}
```

**Design Decisions**:
- Two-pass algorithm: build grid, then detect collisions
- Use atomic operations for thread-safe grid updates
- Cell size = 2 × max radius ensures O(1) neighbor checks
- Elastic collision physics for realistic bouncing
- Damage proportional to collision force (momentum transfer)

### 5. Rendering Manager

**Responsibility**: Render circles using OpenGL instanced rendering with LOD

**Interface**:
```cpp
class RenderingManager {
public:
    bool initialize(uint32_t maxCircles, GLFWwindow* window);
    void render(const PhysicsManager::Circle* circles, uint32_t count, 
                float zoomFactor, const TextureAtlasManager& atlasManager);
    
private:
    GLuint vao_;
    GLuint instanceVBO_;  // Shared with CUDA
    GLuint shaderProgram_;
    cudaGraphicsResource_t cudaVBOResource_;
    
    void setupShaders();
    void setupGeometry();
    uint32_t selectLODLevel(float screenRadius);
};
```

**OpenGL Shader Design**:

Vertex Shader:
```glsl
#version 460 core

layout(location = 0) in vec2 vertexPosition;  // Circle quad vertices
layout(location = 1) in vec2 instancePosition;
layout(location = 2) in float instanceRadius;
layout(location = 3) in vec4 instanceUVs;  // u0, v0, u1, v1
layout(location = 4) in float instanceLOD;

out vec2 fragTexCoord;

uniform mat4 projection;
uniform float zoomFactor;

void main() {
    // Scale vertex by radius and zoom
    vec2 scaledPos = vertexPosition * instanceRadius * zoomFactor;
    vec2 worldPos = instancePosition + scaledPos;
    
    gl_Position = projection * vec4(worldPos, 0.0, 1.0);
    
    // Interpolate UV coordinates
    fragTexCoord = mix(instanceUVs.xy, instanceUVs.zw, (vertexPosition + 1.0) * 0.5);
}
```

Fragment Shader:
```glsl
#version 460 core

in vec2 fragTexCoord;
out vec4 fragColor;

uniform sampler2D atlasTexture;

void main() {
    // Sample texture with automatic LOD selection
    vec4 texColor = texture(atlasTexture, fragTexCoord);
    
    // Discard pixels outside circle (alpha test)
    if (texColor.a < 0.1) discard;
    
    fragColor = texColor;
}
```

**CUDA-OpenGL Interop**:
```cpp
void RenderingManager::updateInstanceData(const PhysicsManager::Circle* circles, uint32_t count) {
    // Map OpenGL VBO to CUDA
    float4* d_instanceData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &cudaVBOResource_, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_instanceData, &numBytes, cudaVBOResource_);
    
    // Launch CUDA kernel to populate instance data
    populateInstanceDataKernel<<<(count + 255) / 256, 256>>>(
        circles, count, d_instanceData, zoomFactor, atlasManager
    );
    
    // Unmap VBO
    cudaGraphicsUnmapResources(1, &cudaVBOResource_, 0);
}

__global__ void populateInstanceDataKernel(
    const Circle* circles,
    uint32_t count,
    float4* instanceData,
    float zoomFactor,
    const TextureAtlasManager& atlasManager
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    const Circle& c = circles[idx];
    const auto& atlasEntry = atlasManager.getAtlasEntry(c.avatarID);
    
    // Calculate screen-space radius for LOD selection
    float screenRadius = c.radius * zoomFactor;
    uint32_t lodLevel = selectLODLevel(screenRadius);
    
    // Pack instance data (position, radius, UVs, LOD)
    instanceData[idx * 3 + 0] = make_float4(c.position.x, c.position.y, c.radius, lodLevel);
    instanceData[idx * 3 + 1] = make_float4(atlasEntry.u0, atlasEntry.v0, atlasEntry.u1, atlasEntry.v1);
    instanceData[idx * 3 + 2] = make_float4(c.health, c.biasMultiplier, 0, 0);  // Extra data
}
```

**LOD Selection Algorithm**:
```cpp
__device__ uint32_t selectLODLevel(float screenRadius) {
    // LOD levels based on screen-space size
    if (screenRadius > 64.0f) return 0;  // Full resolution
    if (screenRadius > 32.0f) return 1;  // 1/2 resolution
    if (screenRadius > 16.0f) return 2;  // 1/4 resolution
    return 3;  // 1/8 resolution
}
```

**Design Decisions**:
- Use instanced rendering (single draw call for all circles)
- Quad geometry (2 triangles) per circle
- CUDA populates instance VBO directly (zero-copy)
- Automatic LOD selection based on screen-space size
- Texture atlas with mipmaps for efficient sampling

### 6. UI Overlay Manager

**Responsibility**: Display performance metrics and game state

**Interface**:
```cpp
class UIManager {
public:
    bool initialize(GLFWwindow* window);
    void render(const PerformanceMetrics& metrics, bool visible);
    void renderWinnerScreen(uint32_t winnerID, const std::string& winnerName);
    
private:
    struct PerformanceMetrics {
        float fps;
        uint32_t activeCircles;
        float cudaKernelTime;
        size_t gpuMemoryUsed;
        size_t systemMemoryUsed;
    };
    
    ImGuiContext* imguiContext_;
    bool metricsVisible_;
};
```

**Design Decisions**:
- Use Dear ImGui for simple overlay rendering
- Toggle visibility with F3 key
- Update metrics every 100ms to avoid overhead
- Display winner screen when simulation ends

## Data Models

### Circle Data Structure

```cpp
struct Circle {
    float2 position;      // 8 bytes
    float2 velocity;      // 8 bytes
    float radius;         // 4 bytes
    float mass;           // 4 bytes
    float health;         // 4 bytes
    uint32_t avatarID;    // 4 bytes
    float biasMultiplier; // 4 bytes
    uint32_t padding;     // 4 bytes (alignment)
};  // Total: 40 bytes per circle
```

**Memory Layout**:
- Structure of Arrays (SoA) would be more cache-friendly, but Array of Structures (AoS) is simpler
- For 1M circles: 40 MB (negligible compared to 12 GB VRAM)
- Aligned to 8 bytes for coalesced memory access

### Spatial Hash Grid

```cpp
struct SpatialCell {
    uint32_t* circleIndices;  // Dynamic array
    uint32_t count;
    uint32_t capacity;
};
```

**Memory Calculation**:
- Arena: 10,000 × 10,000 pixels
- Cell size: 100 pixels (2 × 50 pixel max radius)
- Grid: 100 × 100 = 10,000 cells
- Average circles per cell (1M circles): 100
- Memory: 10,000 cells × 100 indices × 4 bytes = 4 MB

### Texture Atlas

**Format**: BC1 compressed RGBA (4 bits per pixel)
- Atlas size: 8192 × 8192 pixels
- Uncompressed: 256 MB
- BC1 compressed: 64 MB (75% savings)
- Mipmaps (4 levels): 64 + 16 + 4 + 1 = 85 MB total

## Error Handling

### GPU Memory Allocation Failures

```cpp
cudaError_t err = cudaMalloc(&d_circles_, maxCircles * sizeof(Circle));
if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    // Attempt to allocate smaller buffer
    maxCircles /= 2;
    err = cudaMalloc(&d_circles_, maxCircles * sizeof(Circle));
    if (err != cudaSuccess) {
        return false;  // Fatal error
    }
}
```

### Texture Loading Failures

- Skip missing avatar files and use default placeholder texture
- Log warnings for failed loads
- Continue simulation with available textures

### Configuration File Errors

- Validate all parameters on load
- Use default values for missing parameters
- Fail fast with clear error messages for invalid values

## Testing Strategy

### Unit Tests

1. **Configuration Loading**:
   - Test valid configuration files
   - Test invalid/malformed files
   - Test missing files with defaults

2. **Texture Atlas Creation**:
   - Test with various image counts (10, 100, 1000)
   - Test with different image sizes
   - Verify UV coordinate correctness

3. **Physics Calculations**:
   - Test boundary collisions
   - Test circle-circle collisions
   - Verify energy conservation (elastic collisions)

4. **Spatial Hashing**:
   - Test grid construction with known positions
   - Verify collision detection accuracy
   - Test edge cases (circles on cell boundaries)

### Integration Tests

1. **CUDA-OpenGL Interop**:
   - Verify VBO sharing works correctly
   - Test data transfer performance
   - Ensure no memory leaks

2. **End-to-End Simulation**:
   - Run simulation with 1K, 10K, 100K, 1M circles
   - Verify winner is correctly determined
   - Check for memory leaks over long runs

### Performance Tests

1. **Frame Rate Benchmarks**:
   - Measure FPS at 1K, 10K, 100K, 500K, 1M circles
   - Target: 60+ FPS up to 100K circles
   - Target: 30+ FPS at 500K circles
   - Target: 15+ FPS at 1M circles

2. **Memory Usage**:
   - Verify GPU memory usage stays under 10 GB
   - Check for memory leaks (run for 10 minutes)

3. **Kernel Performance**:
   - Profile CUDA kernels with nvprof/Nsight Compute
   - Identify bottlenecks
   - Optimize hot paths

### Stress Tests

1. **Maximum Circle Count**:
   - Test with 1M circles for stability
   - Verify no crashes or hangs

2. **Long-Running Simulation**:
   - Run until only one circle remains
   - Verify correct winner determination

3. **Configuration Hot-Reload**:
   - Modify bias.txt and damage_scaling.txt during runtime
   - Verify changes take effect within 1 second

## Performance Optimization Strategies

### Memory Access Patterns

1. **Coalesced Memory Access**:
   - Ensure threads in a warp access consecutive memory addresses
   - Use aligned data structures (8-byte alignment)

2. **Shared Memory Usage**:
   - Cache frequently accessed data in shared memory
   - Use shared memory for collision detection within a block

3. **Texture Memory**:
   - Use CUDA texture memory for avatar sampling (automatic caching)
   - Leverage 2D spatial locality for better cache hit rates

### Kernel Optimization

1. **Occupancy**:
   - Target 50%+ occupancy (warps per SM)
   - Balance register usage vs. occupancy
   - Use `__launch_bounds__` to control register allocation

2. **Divergence Minimization**:
   - Minimize branch divergence within warps
   - Use warp-level primitives (`__ballot_sync`, `__shfl_sync`)

3. **Atomic Operations**:
   - Minimize atomic operations (use local accumulation when possible)
   - Use `atomicAdd` for damage application (unavoidable)

### Rendering Optimization

1. **Instanced Rendering**:
   - Single draw call for all circles
   - Minimize state changes

2. **Texture Compression**:
   - Use BC1 compression for 75% memory savings
   - Reduces memory bandwidth requirements

3. **LOD System**:
   - Automatic mipmap selection reduces texture sampling cost
   - Smaller circles use lower resolution textures

## References

- NVIDIA CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- GPU Gems 3 - Chapter 31 (N-body Simulation): https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
- Spatial Hashing for Collision Detection: https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
- OpenGL Instanced Rendering: https://www.khronos.org/opengl/wiki/Vertex_Rendering#Instancing
- CUDA-OpenGL Interoperability: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html
- BC Texture Compression: https://www.khronos.org/opengl/wiki/S3_Texture_Compression
- RTX 3060 Specifications: https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-mobile.c3682
