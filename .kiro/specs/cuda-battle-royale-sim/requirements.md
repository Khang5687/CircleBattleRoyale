# Requirements Document

## Introduction

This document specifies the requirements for a high-performance 2D battle royale simulation system capable of handling up to 1 million circles (players) with real-time physics, collision detection, and image-based rendering. The system is optimized for NVIDIA RTX 3060 laptop GPU (3000 series) using CUDA for physics computation and OpenGL/Vulkan for rendering. Each circle represents a player with a unique avatar image loaded from disk, and the simulation features dynamic LOD-based rendering to maintain performance at scale.

## Glossary

- **System**: The complete 2D battle royale simulation application
- **Circle**: A circular entity representing a player in the simulation
- **Avatar**: A JPG/PNG image file representing a player's visual appearance
- **LOD (Level of Detail)**: A rendering optimization technique that reduces image quality based on display size
- **Mipmap**: Pre-calculated, optimized sequences of images at progressively lower resolutions
- **Texture Atlas**: A single large texture containing multiple smaller textures (avatars)
- **Spatial Hashing**: A collision detection optimization technique that divides space into a grid
- **CUDA**: NVIDIA's parallel computing platform for GPU acceleration
- **VBO (Vertex Buffer Object)**: OpenGL buffer for storing vertex data
- **CUDA-OpenGL Interop**: Technology allowing CUDA and OpenGL to share memory
- **Bias System**: A mechanism to increase specific players' chances of winning
- **Damage Scaling**: Dynamic damage adjustment based on remaining player count
- **Performance Metrics**: Real-time statistics about simulation performance (FPS, particle count, etc.)

## Requirements

### Requirement 1: High-Performance Physics Simulation

**User Story:** As a simulation operator, I want the system to handle up to 1 million circles with real-time physics, so that I can observe large-scale battle dynamics.

#### Acceptance Criteria

1. WHEN the System initializes, THE System SHALL allocate CUDA memory for up to 1,000,000 circle positions and velocities
2. WHILE the simulation is running, THE System SHALL update all circle positions and velocities at a minimum rate of 60 frames per second for up to 100,000 circles
3. WHEN two circles collide, THE System SHALL calculate bounce physics using elastic collision formulas within 1 millisecond
4. THE System SHALL implement spatial hashing with a grid cell size optimized for the average circle radius to achieve O(n) collision detection complexity
5. WHILE processing collisions, THE System SHALL use CUDA shared memory for tile-based processing to minimize global memory access latency

### Requirement 2: Collision Detection and Damage System

**User Story:** As a simulation operator, I want circles to detect collisions and apply damage to each other, so that the battle royale progresses toward a winner.

#### Acceptance Criteria

1. WHEN two circles collide, THE System SHALL calculate the collision force based on relative velocities and masses
2. WHEN a collision occurs, THE System SHALL apply damage to both circles proportional to the collision force
3. WHILE a circle's health is above zero, THE System SHALL maintain the circle in the active simulation
4. WHEN a circle's health reaches zero, THE System SHALL remove the circle from the simulation within one frame
5. THE System SHALL track the remaining circle count and declare a winner when only one circle remains

### Requirement 3: Image-Based Rendering with LOD

**User Story:** As a simulation operator, I want each circle to display a unique avatar image with automatic quality adjustment, so that the simulation remains performant while maintaining visual fidelity.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL load all JPG/PNG images from the assets/players directory into a CUDA texture atlas
2. WHEN loading images, THE System SHALL generate mipmaps with at least 4 LOD levels for each avatar
3. WHILE rendering circles, THE System SHALL select the appropriate mipmap level based on the circle's screen-space radius
4. WHEN the circle count increases, THE System SHALL automatically reduce circle sizes to fit all circles on screen
5. THE System SHALL use CUDA texture memory with 2D spatial locality for efficient texture sampling during rendering

### Requirement 4: Dynamic Scaling and Zoom

**User Story:** As a simulation operator, I want circles to automatically scale based on population and allow manual zoom adjustment, so that I can observe the simulation at different detail levels.

#### Acceptance Criteria

1. WHEN the circle count exceeds 10,000, THE System SHALL reduce the base circle radius by a factor inversely proportional to the square root of the circle count
2. WHILE the user presses the zoom in key, THE System SHALL increase the zoom factor by 10% per second up to a maximum of 500%
3. WHILE the user presses the zoom out key, THE System SHALL decrease the zoom factor by 10% per second down to a minimum of 10%
4. THE System SHALL combine the automatic scaling factor with the user zoom factor to determine final circle display size
5. WHEN the zoom factor changes, THE System SHALL update the LOD selection algorithm to maintain appropriate texture quality

### Requirement 5: Bias System for Win Probability

**User Story:** As a simulation operator, I want to configure bias values for specific players, so that I can influence their chances of winning the battle royale.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL read a bias configuration file (bias.txt) containing player names and bias multipliers
2. WHERE a player has a bias multiplier greater than 1.0, THE System SHALL increase that circle's initial health by the bias multiplier
3. WHERE a player has a bias multiplier greater than 1.0, THE System SHALL reduce damage received by dividing incoming damage by the bias multiplier
4. THE System SHALL apply bias values without affecting the physics simulation (mass, velocity, collision detection)
5. WHEN the bias file is modified during runtime, THE System SHALL reload bias values within 1 second

### Requirement 6: Damage Scaling Based on Population

**User Story:** As a simulation operator, I want damage to scale based on remaining player count, so that the simulation pace adjusts dynamically.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL read a damage scaling configuration file (damage_scaling.txt) containing population thresholds and damage multipliers
2. WHEN the remaining circle count crosses a threshold defined in the configuration, THE System SHALL update the global damage multiplier within one frame
3. WHILE applying collision damage, THE System SHALL multiply the base damage by the current damage multiplier
4. THE System SHALL support at least 10 different population thresholds with independent damage multipliers
5. WHEN the damage scaling file is modified during runtime, THE System SHALL reload scaling values within 1 second

### Requirement 7: Performance Metrics Display

**User Story:** As a simulation operator, I want to view real-time performance metrics, so that I can monitor system performance and optimize settings.

#### Acceptance Criteria

1. WHEN the user presses the F3 key, THE System SHALL toggle the visibility of the performance metrics overlay
2. WHILE the performance overlay is visible, THE System SHALL display the current frames per second (FPS) updated every 100 milliseconds
3. WHILE the performance overlay is visible, THE System SHALL display the current active circle count updated every frame
4. WHILE the performance overlay is visible, THE System SHALL display the current CUDA kernel execution time in milliseconds
5. WHILE the performance overlay is visible, THE System SHALL display the current memory usage (GPU and system RAM) in megabytes

### Requirement 8: CUDA-OpenGL Interoperability

**User Story:** As a system developer, I want efficient data sharing between CUDA and OpenGL, so that rendering performance is maximized.

#### Acceptance Criteria

1. WHEN the System initializes rendering, THE System SHALL create OpenGL VBOs registered with CUDA for zero-copy memory access
2. WHILE updating physics, THE System SHALL write circle positions directly to CUDA-mapped VBO memory
3. WHILE rendering, THE System SHALL use OpenGL instanced rendering to draw all circles in a single draw call
4. THE System SHALL maintain separate CUDA streams for physics computation and rendering preparation to enable concurrent execution
5. WHEN transferring texture data, THE System SHALL use CUDA texture memory bound to OpenGL textures to avoid CPU-GPU transfers

### Requirement 9: Optimized Memory Management

**User Story:** As a system developer, I want efficient memory allocation and management, so that the system can handle maximum circle counts without running out of memory.

#### Acceptance Criteria

1. WHEN the System allocates CUDA memory, THE System SHALL use pinned (page-locked) host memory for faster CPU-GPU transfers
2. THE System SHALL allocate a single contiguous CUDA memory block for all circle data (positions, velocities, health) to minimize allocation overhead
3. WHEN loading avatar images, THE System SHALL compress the texture atlas using BC1/BC3 compression to reduce GPU memory usage by at least 75%
4. THE System SHALL implement a memory pool for temporary collision detection buffers to avoid repeated allocations
5. WHEN the circle count drops below 50% of the initial count, THE System SHALL optionally compact memory to free unused GPU memory

### Requirement 10: Configuration and Initialization

**User Story:** As a simulation operator, I want to configure simulation parameters before starting, so that I can customize the battle royale behavior.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL read a configuration file (config.txt) containing initial circle count, arena size, and physics parameters
2. THE System SHALL validate that the initial circle count does not exceed 1,000,000 and the arena size is at least 1000x1000 pixels
3. WHEN configuration validation fails, THE System SHALL display an error message and terminate within 1 second
4. THE System SHALL support command-line arguments to override configuration file values
5. WHEN the System initializes successfully, THE System SHALL display a startup message showing the loaded configuration parameters

## References

- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA GPU Gems Chapter 31 (N-body Simulation): https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
- CUDA-OpenGL Interoperability: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html
- Spatial Hashing for Collision Detection: https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
- OpenGL Instanced Rendering: https://www.khronos.org/opengl/wiki/Vertex_Rendering#Instancing
