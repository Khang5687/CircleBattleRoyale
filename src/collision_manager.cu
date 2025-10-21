#include "collision_manager.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA kernel to build spatial hash grid
__global__ void buildSpatialHashKernel(
    const PhysicsManager::Circle* circles,
    uint32_t count,
    uint32_t* gridCounts,
    uint32_t* gridIndices,
    uint32_t gridWidth,
    uint32_t gridHeight,
    float cellSize,
    uint32_t maxCirclesPerCell
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    const PhysicsManager::Circle& c = circles[idx];
    
    // Calculate grid cell coordinates
    uint32_t cellX = static_cast<uint32_t>(c.position.x / cellSize);
    uint32_t cellY = static_cast<uint32_t>(c.position.y / cellSize);
    
    // Clamp to grid bounds
    cellX = min(cellX, gridWidth - 1);
    cellY = min(cellY, gridHeight - 1);
    
    uint32_t cellIdx = cellY * gridWidth + cellX;
    
    // Atomically increment cell count and get slot index
    uint32_t slot = atomicAdd(&gridCounts[cellIdx], 1);
    
    // Store circle index in grid (if there's space)
    if (slot < maxCirclesPerCell) {
        gridIndices[cellIdx * maxCirclesPerCell + slot] = idx;
    }
}

bool CollisionManager::initialize(uint32_t maxCircles, float arenaWidth, float arenaHeight, float cellSize) {
    cellSize_ = cellSize;
    
    // Calculate grid dimensions
    gridWidth_ = static_cast<uint32_t>(std::ceil(arenaWidth / cellSize)) + 1;
    gridHeight_ = static_cast<uint32_t>(std::ceil(arenaHeight / cellSize)) + 1;
    
    uint32_t totalCells = gridWidth_ * gridHeight_;
    
    // Estimate max circles per cell (average + buffer)
    // For 1M circles and ~10K cells, average is 100 circles/cell
    // Use 200 as buffer for uneven distribution
    uint32_t maxCirclesPerCell = 200;
    
    std::cout << "CollisionManager initializing spatial hash grid:" << std::endl;
    std::cout << "  Grid dimensions: " << gridWidth_ << " x " << gridHeight_ << " = " << totalCells << " cells" << std::endl;
    std::cout << "  Cell size: " << cellSize << std::endl;
    std::cout << "  Max circles per cell: " << maxCirclesPerCell << std::endl;
    
    // Allocate grid memory on device
    // We need: counts array + indices array
    uint32_t* d_gridCounts;
    uint32_t* d_gridIndices;
    
    CUDA_CHECK(cudaMalloc(&d_gridCounts, totalCells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_gridIndices, totalCells * maxCirclesPerCell * sizeof(uint32_t)));
    
    // Allocate SpatialCell array
    CUDA_CHECK(cudaMalloc(&d_grid_, totalCells * sizeof(SpatialCell)));
    
    // Initialize SpatialCell structures on host, then copy to device
    SpatialCell* h_grid = new SpatialCell[totalCells];
    for (uint32_t i = 0; i < totalCells; i++) {
        h_grid[i].circleIndices = d_gridIndices + (i * maxCirclesPerCell);
        h_grid[i].count = 0;
        h_grid[i].capacity = maxCirclesPerCell;
    }
    
    CUDA_CHECK(cudaMemcpy(d_grid_, h_grid, totalCells * sizeof(SpatialCell), cudaMemcpyHostToDevice));
    delete[] h_grid;
    
    // Create CUDA stream for collision operations
    CUDA_CHECK(cudaStreamCreate(&collisionStream_));
    
    std::cout << "CollisionManager initialized successfully" << std::endl;
    std::cout << "  Total grid memory: " << (totalCells * sizeof(SpatialCell) + totalCells * sizeof(uint32_t) + totalCells * maxCirclesPerCell * sizeof(uint32_t)) / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

// CUDA kernel for collision detection with shared memory optimization
__global__ void detectCollisionsKernel(
    PhysicsManager::Circle* circles,
    uint32_t count,
    const uint32_t* gridCounts,
    const uint32_t* gridIndices,
    uint32_t gridWidth,
    uint32_t gridHeight,
    float cellSize,
    uint32_t maxCirclesPerCell,
    float damageMultiplier
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    PhysicsManager::Circle& c1 = circles[idx];
    
    // Calculate grid cell for this circle
    uint32_t cellX = static_cast<uint32_t>(c1.position.x / cellSize);
    uint32_t cellY = static_cast<uint32_t>(c1.position.y / cellSize);
    
    // Clamp to grid bounds
    cellX = min(cellX, gridWidth - 1);
    cellY = min(cellY, gridHeight - 1);
    
    // Check 3x3 neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = static_cast<int>(cellX) + dx;
            int ny = static_cast<int>(cellY) + dy;
            
            // Check bounds
            if (nx < 0 || nx >= static_cast<int>(gridWidth) || 
                ny < 0 || ny >= static_cast<int>(gridHeight)) {
                continue;
            }
            
            uint32_t cellIdx = ny * gridWidth + nx;
            uint32_t cellCount = gridCounts[cellIdx];
            
            // Iterate through circles in this cell
            for (uint32_t i = 0; i < cellCount && i < maxCirclesPerCell; i++) {
                uint32_t otherIdx = gridIndices[cellIdx * maxCirclesPerCell + i];
                
                // Skip self and avoid duplicate checks
                if (otherIdx <= idx) continue;
                if (otherIdx >= count) continue;
                
                PhysicsManager::Circle& c2 = circles[otherIdx];
                
                // Circle-circle collision test
                float dx_val = c2.position.x - c1.position.x;
                float dy_val = c2.position.y - c1.position.y;
                float distSq = dx_val * dx_val + dy_val * dy_val;
                float minDist = c1.radius + c2.radius;
                float minDistSq = minDist * minDist;
                
                if (distSq < minDistSq && distSq > 1e-6f) {
                    // Collision detected
                    float dist = sqrtf(distSq);
                    float nx_val = dx_val / dist;
                    float ny_val = dy_val / dist;
                    
                    // Relative velocity
                    float dvx = c2.velocity.x - c1.velocity.x;
                    float dvy = c2.velocity.y - c1.velocity.y;
                    float dvn = dvx * nx_val + dvy * ny_val;
                    
                    // Only process if circles are moving toward each other
                    if (dvn < 0.0f) {
                        // Elastic collision impulse
                        float impulse = 2.0f * dvn / (c1.mass + c2.mass);
                        
                        // Apply velocity changes
                        float impulse1x = impulse * c2.mass * nx_val;
                        float impulse1y = impulse * c2.mass * ny_val;
                        float impulse2x = -impulse * c1.mass * nx_val;
                        float impulse2y = -impulse * c1.mass * ny_val;
                        
                        atomicAdd(&c1.velocity.x, impulse1x);
                        atomicAdd(&c1.velocity.y, impulse1y);
                        atomicAdd(&c2.velocity.x, impulse2x);
                        atomicAdd(&c2.velocity.y, impulse2y);
                        
                        // Calculate collision force for damage
                        float collisionForce = fabsf(dvn) * (c1.mass + c2.mass);
                        float baseDamage = collisionForce * 0.01f;  // Tunable constant
                        
                        // Apply damage with scaling and bias
                        float damage1 = baseDamage * damageMultiplier / c1.biasMultiplier;
                        float damage2 = baseDamage * damageMultiplier / c2.biasMultiplier;
                        
                        atomicAdd(&c1.health, -damage1);
                        atomicAdd(&c2.health, -damage2);
                    }
                }
            }
        }
    }
}

void CollisionManager::detectAndResolveCollisions(PhysicsManager::Circle* circles, uint32_t count, float damageMultiplier) {
    if (count == 0) return;
    
    uint32_t totalCells = gridWidth_ * gridHeight_;
    uint32_t maxCirclesPerCell = 200;  // Must match initialization
    
    // Get device pointers for counts and indices
    uint32_t* d_gridCounts;
    uint32_t* d_gridIndices;
    
    cudaMalloc(&d_gridCounts, totalCells * sizeof(uint32_t));
    cudaMalloc(&d_gridIndices, totalCells * maxCirclesPerCell * sizeof(uint32_t));
    
    // Clear grid counts
    cudaMemset(d_gridCounts, 0, totalCells * sizeof(uint32_t));
    
    // Build spatial hash grid
    const int threadsPerBlock = 256;
    const int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    buildSpatialHashKernel<<<numBlocks, threadsPerBlock, 0, collisionStream_>>>(
        circles,
        count,
        d_gridCounts,
        d_gridIndices,
        gridWidth_,
        gridHeight_,
        cellSize_,
        maxCirclesPerCell
    );
    
    // Detect and resolve collisions
    detectCollisionsKernel<<<numBlocks, threadsPerBlock, 0, collisionStream_>>>(
        circles,
        count,
        d_gridCounts,
        d_gridIndices,
        gridWidth_,
        gridHeight_,
        cellSize_,
        maxCirclesPerCell,
        damageMultiplier
    );
    
    // Synchronize stream
    cudaStreamSynchronize(collisionStream_);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA collision kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Free temporary memory
    cudaFree(d_gridCounts);
    cudaFree(d_gridIndices);
}

void CollisionManager::cleanup() {
    if (d_grid_) {
        // First, we need to free the individual circleIndices arrays
        // Copy grid to host to get the pointers
        uint32_t totalCells = gridWidth_ * gridHeight_;
        SpatialCell* h_grid = new SpatialCell[totalCells];
        cudaMemcpy(h_grid, d_grid_, totalCells * sizeof(SpatialCell), cudaMemcpyDeviceToHost);
        
        // Free the first circleIndices pointer (they're all in one contiguous block)
        if (totalCells > 0 && h_grid[0].circleIndices) {
            cudaFree(h_grid[0].circleIndices);
        }
        
        delete[] h_grid;
        
        // Free the grid itself
        cudaFree(d_grid_);
        d_grid_ = nullptr;
    }
    
    if (collisionStream_) {
        cudaStreamDestroy(collisionStream_);
        collisionStream_ = nullptr;
    }
}
