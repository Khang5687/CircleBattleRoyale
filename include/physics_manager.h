#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "config_manager.h"

class PhysicsManager {
public:
    struct Circle {
        float2 position;
        float2 velocity;
        float radius;
        float mass;
        float health;
        uint32_t avatarID;
        float biasMultiplier;
        uint32_t padding;  // Alignment
    };
    
    bool initialize(uint32_t maxCircles, const ConfigManager::SimConfig& config);
    void updatePhysics(float deltaTime);
    void applyBoundaryConditions(float arenaWidth, float arenaHeight);
    
    Circle* getCircleData() { return d_circles_; }
    uint32_t getActiveCircleCount() const { return activeCount_; }
    
    void cleanup();
    
private:
    Circle* d_circles_;  // Device memory
    uint32_t maxCircles_;
    uint32_t activeCount_;
    cudaStream_t physicsStream_;
};
