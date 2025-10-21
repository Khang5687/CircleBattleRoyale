#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "physics_manager.h"

class CollisionManager {
public:
    bool initialize(uint32_t maxCircles, float arenaWidth, float arenaHeight, float cellSize);
    void detectAndResolveCollisions(PhysicsManager::Circle* circles, uint32_t count, float damageMultiplier);
    
    void cleanup();
    
private:
    struct SpatialCell {
        uint32_t* circleIndices;
        uint32_t count;
        uint32_t capacity;
    };
    
    SpatialCell* d_grid_;
    uint32_t gridWidth_;
    uint32_t gridHeight_;
    float cellSize_;
    cudaStream_t collisionStream_;
};
