# Implementation Plan

- [x] 1. Set up project structure and development environment


  - Create CMake build system with CUDA, OpenGL, and GLFW dependencies
  - Configure compiler flags for compute capability 8.6 (RTX 3060)
  - Set up directory structure: src/, include/, shaders/, assets/, config/
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 2. Implement configuration management system

  - [ ] 2.1 Create ConfigManager class with file parsing

    - Implement loadConfig() to parse config.txt (circle count, arena size, physics params)
    - Implement loadBias() to parse bias.txt (player name, bias multiplier)
    - Implement loadDamageScaling() to parse damage_scaling.txt (threshold, multiplier)
    - Add validation for all parameters (ranges, required fields)
    - _Requirements: 10.1, 10.2, 10.4, 5.1, 6.1_

  - [ ] 2.2 Implement hot-reload system for runtime configuration
    - Use file watching (inotify on Linux, ReadDirectoryChangesW on Windows)
    - Reload bias and damage scaling files when modified
    - Apply changes within 1 second without restarting simulation
    - _Requirements: 5.5, 6.5_

- [ ] 3. Implement texture atlas system with ASTC compression

  - [ ] 3.1 Create image loading pipeline

    - Use stb_image to load JPG/PNG files from assets/players directory
    - Implement error handling for missing/corrupt files
    - Create placeholder texture for failed loads
    - _Requirements: 3.1, 9.3_

  - [ ] 3.2 Implement texture atlas packing

    - Create simple row-based packing algorithm (8192x8192 atlas)
    - Generate UV coordinates for each avatar
    - Support multiple atlases if >2000 avatars
    - _Requirements: 3.1, 3.5_

  - [ ] 3.3 Implement ASTC compression on GPU

    - Write CUDA kernel for ASTC 8x8 block compression
    - Generate 4 mipmap levels (full, 1/2, 1/4, 1/8 resolution)
    - Upload compressed data to CUDA texture memory
    - _Requirements: 3.2, 3.5, 9.3_

  - [ ] 3.4 Implement streaming texture manager
    - Create LRU cache for 2000 most recently used textures
    - Implement on-demand texture loading based on visible circles
    - Add eviction policy when memory pressure is high
    - _Requirements: 3.1, 9.1, 9.4_

- [ ] 4. Implement CUDA physics simulation

  - [ ] 4.1 Create Circle data structure and memory allocation

    - Define Circle struct with FP16 positions, FP32 critical data
    - Allocate pinned host memory and CUDA device memory
    - Initialize circles with random positions, velocities, health
    - Apply bias multipliers from configuration
    - _Requirements: 1.1, 5.2, 9.1, 9.2_

  - [ ] 4.2 Implement physics update kernel

    - Write updatePhysicsKernel() for Euler integration
    - Update positions based on velocities and deltaTime
    - Apply boundary collision detection and elastic bounce
    - Use 256 threads per block for optimal occupancy
    - _Requirements: 1.2, 1.3, 1.4_

  - [ ] 4.3 Implement dynamic scaling system
    - Calculate scale factor based on circle count (inversely proportional to sqrt(count))
    - Combine with user zoom factor (10-500%)
    - Update circle radii for rendering
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. Implement spatial hashing collision detection

  - [ ] 5.1 Create spatial hash grid structure

    - Calculate optimal cell size (2 × max circle radius)
    - Allocate grid memory (width × height cells)
    - Implement grid construction kernel with atomic operations
    - _Requirements: 1.4, 2.1, 2.2_

  - [ ] 5.2 Implement collision detection with shared memory

    - Write detectCollisionsOptimized() kernel using shared memory
    - Load 3x3 neighborhood into shared memory for fast access
    - Perform circle-circle intersection tests
    - Apply elastic collision physics (momentum transfer)
    - _Requirements: 1.3, 2.1, 2.2, 2.3_

  - [ ] 5.3 Implement damage application system

    - Calculate collision force based on relative velocities
    - Apply damage proportional to force and damage multiplier
    - Use atomic operations for thread-safe health updates
    - Apply bias multiplier to reduce damage for biased players
    - _Requirements: 2.2, 2.3, 5.3, 6.2, 6.3_

  - [ ] 5.4 Implement health check and circle removal
    - Write kernel to mark dead circles (health <= 0)
    - Implement parallel compaction to remove dead circles
    - Update active circle count
    - Declare winner when only one circle remains
    - _Requirements: 2.3, 2.4, 2.5_

- [ ] 6. Implement GPU-driven rendering system

  - [ ] 6.1 Set up OpenGL context and window

    - Initialize GLFW window with OpenGL 4.6 context
    - Configure viewport and projection matrix
    - Enable depth testing and alpha blending
    - _Requirements: 8.1, 8.2_

  - [ ] 6.2 Implement CUDA-OpenGL interop

    - Create OpenGL VBO and register with CUDA
    - Implement zero-copy memory mapping
    - Set up CUDA streams for concurrent execution
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 6.3 Implement frustum culling compute shader

    - Write CUDA kernel to test circles against frustum planes
    - Build list of visible circle indices
    - Update indirect draw command buffer
    - _Requirements: 3.3, 3.4, 8.3_

  - [ ] 6.4 Create vertex and fragment shaders

    - Write vertex shader for instanced rendering with LOD
    - Implement automatic LOD selection based on screen-space radius
    - Write fragment shader with texture sampling and alpha test
    - _Requirements: 3.3, 3.4, 3.5_

  - [ ] 6.5 Implement instanced rendering pipeline
    - Set up quad geometry (2 triangles per circle)
    - Populate instance VBO with circle data (position, radius, UVs)
    - Use glMultiDrawElementsIndirect for single draw call
    - _Requirements: 8.2, 8.3, 8.4_

- [ ] 7. Implement performance metrics and UI overlay

  - [ ] 7.1 Set up Dear ImGui integration

    - Initialize ImGui with OpenGL backend
    - Create performance metrics overlay
    - Implement F3 toggle for visibility
    - _Requirements: 7.1, 7.2_

  - [ ] 7.2 Implement performance tracking

    - Measure FPS using frame time averaging (100ms window)
    - Track active circle count
    - Profile CUDA kernel execution time using events
    - Query GPU and system memory usage
    - _Requirements: 7.2, 7.3, 7.4, 7.5_

  - [ ] 7.3 Create winner announcement screen
    - Display winner ID and name when simulation ends
    - Show final statistics (total time, collisions, etc.)
    - _Requirements: 2.5_

- [ ] 8. Implement input handling and user controls

  - [ ] 8.1 Create input manager

    - Handle keyboard input (F3 for metrics, +/- for zoom)
    - Implement zoom factor adjustment (10-500%)
    - Add pause/resume functionality
    - _Requirements: 4.2, 4.3, 7.1_

  - [ ] 8.2 Implement camera controls
    - Add pan and zoom with mouse
    - Implement smooth camera interpolation
    - _Requirements: 4.4, 4.5_

- [ ] 9. Implement asynchronous texture loading

  - [ ] 9.1 Create thread pool for CPU image loading

    - Implement worker threads for parallel image loading
    - Use producer-consumer queue for work distribution
    - _Requirements: 3.1, 9.1_

  - [ ] 9.2 Implement CUDA stream-based async upload
    - Create multiple CUDA streams for concurrent uploads
    - Overlap CPU loading, GPU compression, and memory transfer
    - Implement progressive loading (start simulation before all textures loaded)
    - _Requirements: 8.4, 8.5, 9.1_

- [ ] 10. Implement memory management and optimization

  - [ ] 10.1 Create memory pool for temporary buffers

    - Implement reusable buffer pool for collision detection
    - Avoid repeated allocations/deallocations
    - _Requirements: 9.4_

  - [ ] 10.2 Implement memory compaction

    - Compact circle array when count drops below 50%
    - Free unused GPU memory
    - _Requirements: 9.5_

  - [ ] 10.3 Add memory usage monitoring
    - Track CUDA memory allocations
    - Warn when approaching 5.5 GB limit
    - _Requirements: 7.5, 9.1_

- [ ] 11. Integration and main simulation loop

  - [ ] 11.1 Implement main simulation loop

    - Initialize all subsystems (config, textures, physics, rendering)
    - Run simulation loop at 60+ FPS target
    - Handle frame timing and deltaTime calculation
    - _Requirements: 1.2, 10.5_

  - [ ] 11.2 Wire up all components

    - Connect physics updates to collision detection
    - Link collision detection to damage application
    - Connect visible circles to rendering pipeline
    - Update UI overlay with performance metrics
    - _Requirements: All requirements_

  - [ ] 11.3 Implement graceful shutdown
    - Clean up CUDA resources (memory, streams, textures)
    - Clean up OpenGL resources (VBOs, shaders, textures)
    - Display final statistics
    - _Requirements: 10.5_

- [ ]\* 12. Testing and validation

  - [ ]\* 12.1 Write unit tests for configuration loading

    - Test valid and invalid config files
    - Test bias and damage scaling parsing
    - Verify default values for missing parameters
    - _Requirements: 10.1, 10.2_

  - [ ]\* 12.2 Write unit tests for texture atlas

    - Test with various image counts (10, 100, 1000)
    - Verify UV coordinate correctness
    - Test ASTC compression quality
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]\* 12.3 Write unit tests for physics calculations

    - Test boundary collisions
    - Test circle-circle collisions
    - Verify energy conservation
    - _Requirements: 1.3, 2.1, 2.2_

  - [ ]\* 12.4 Write integration tests

    - Test CUDA-OpenGL interop
    - Test end-to-end simulation with 1K, 10K, 100K circles
    - Verify winner determination
    - Check for memory leaks
    - _Requirements: 8.1, 8.2, 2.5_

  - [ ]\* 12.5 Run performance benchmarks

    - Measure FPS at 1K, 10K, 100K, 500K, 1M circles
    - Profile CUDA kernels with Nsight Compute
    - Verify memory usage stays under 5.5 GB
    - _Requirements: 1.2, 7.2, 9.1_

  - [ ]\* 12.6 Run stress tests
    - Test with 1M circles for stability
    - Run until only one circle remains
    - Test configuration hot-reload
    - _Requirements: 1.1, 5.5, 6.5_

- [ ]\* 13. Documentation and polish

  - [ ]\* 13.1 Write user documentation

    - Create README with build instructions
    - Document configuration file formats
    - Add usage examples
    - _Requirements: 10.4_

  - [ ]\* 13.2 Write developer documentation

    - Document code architecture
    - Add inline comments for complex algorithms
    - Create performance tuning guide
    - _Requirements: All requirements_

  - [ ]\* 13.3 Create sample configuration files
    - Provide example config.txt with reasonable defaults
    - Create sample bias.txt with example players
    - Create sample damage_scaling.txt with balanced values
    - _Requirements: 10.1, 5.1, 6.1_
