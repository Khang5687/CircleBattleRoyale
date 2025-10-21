#include "physics_manager.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
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

bool PhysicsManager::initialize(uint32_t maxCircles, const ConfigManager::SimConfig& config) {
    maxCircles_ = maxCircles;
    activeCount_ = config.initialCircleCount;
    
    // Cache config parameters
    arenaWidth_ = config.arenaWidth;
    arenaHeight_ = config.arenaHeight;
    elasticity_ = config.elasticity;
    baseRadius_ = config.baseCircleRadius;
    
    // Allocate pinned host memory for faster CPU-GPU transfers
    Circle* h_circles;
    CUDA_CHECK(cudaMallocHost(&h_circles, maxCircles * sizeof(Circle)));
    
    // Initialize circles with random positions and velocities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> posX(config.baseCircleRadius, config.arenaWidth - config.baseCircleRadius);
    std::uniform_real_distribution<float> posY(config.baseCircleRadius, config.arenaHeight - config.baseCircleRadius);
    std::uniform_real_distribution<float> vel(-100.0f, 100.0f);
    
    for (uint32_t i = 0; i < activeCount_; i++) {
        h_circles[i].position.x = posX(gen);
        h_circles[i].position.y = posY(gen);
        h_circles[i].velocity.x = vel(gen);
        h_circles[i].velocity.y = vel(gen);
        h_circles[i].radius = config.baseCircleRadius;
        h_circles[i].mass = config.baseCircleMass;
        h_circles[i].health = config.baseHealth;
        h_circles[i].avatarID = i % 1000;  // Cycle through avatar IDs (will be updated when texture system is integrated)
        h_circles[i].biasMultiplier = 1.0f;  // Default, will be updated based on bias config
        h_circles[i].padding = 0;
    }
    
    // Allocate CUDA device memory
    CUDA_CHECK(cudaMalloc(&d_circles_, maxCircles * sizeof(Circle)));
    
    // Copy initialized data to device
    CUDA_CHECK(cudaMemcpy(d_circles_, h_circles, activeCount_ * sizeof(Circle), cudaMemcpyHostToDevice));
    
    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(h_circles));
    
    // Create CUDA stream for physics operations
    CUDA_CHECK(cudaStreamCreate(&physicsStream_));
    
    std::cout << "PhysicsManager initialized with " << activeCount_ << " circles" << std::endl;
    std::cout << "  Arena: " << config.arenaWidth << "x" << config.arenaHeight << std::endl;
    std::cout << "  Base radius: " << config.baseCircleRadius << std::endl;
    std::cout << "  Base mass: " << config.baseCircleMass << std::endl;
    std::cout << "  Base health: " << config.baseHealth << std::endl;
    
    return true;
}

void PhysicsManager::applyBiasMultipliers(const std::vector<ConfigManager::BiasEntry>& biasEntries) {
    if (biasEntries.empty()) {
        return;
    }
    
    // Copy circles to host to apply bias
    Circle* h_circles;
    cudaMallocHost(&h_circles, activeCount_ * sizeof(Circle));
    cudaMemcpy(h_circles, d_circles_, activeCount_ * sizeof(Circle), cudaMemcpyDeviceToHost);
    
    // Apply bias multipliers based on player names
    // For now, we'll apply bias to circles based on their avatar ID matching the bias entry index
    for (const auto& biasEntry : biasEntries) {
        // Simple mapping: use hash of player name to determine which circles get the bias
        std::hash<std::string> hasher;
        size_t hash = hasher(biasEntry.playerName);
        
        // Apply bias to circles whose ID matches this hash modulo active count
        for (uint32_t i = 0; i < activeCount_; i++) {
            if ((hash % activeCount_) == i) {
                h_circles[i].biasMultiplier = biasEntry.biasMultiplier;
                h_circles[i].health *= biasEntry.biasMultiplier;  // Increase initial health
                std::cout << "Applied bias " << biasEntry.biasMultiplier 
                          << " to circle " << i << " (player: " << biasEntry.playerName << ")" << std::endl;
                break;  // Only apply to one circle per bias entry
            }
        }
    }
    
    // Copy back to device
    cudaMemcpy(d_circles_, h_circles, activeCount_ * sizeof(Circle), cudaMemcpyHostToDevice);
    cudaFreeHost(h_circles);
}

// CUDA kernel for physics update with Euler integration and boundary collisions
__global__ void updatePhysicsKernel(
    PhysicsManager::Circle* circles,
    uint32_t count,
    float deltaTime,
    float arenaWidth,
    float arenaHeight,
    float elasticity
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    PhysicsManager::Circle& c = circles[idx];
    
    // Update position using Euler integration
    c.position.x += c.velocity.x * deltaTime;
    c.position.y += c.velocity.y * deltaTime;
    
    // Boundary collision detection and elastic bounce
    bool collided = false;
    
    // Left boundary
    if (c.position.x - c.radius < 0.0f) {
        c.position.x = c.radius;
        c.velocity.x = -c.velocity.x * elasticity;
        collided = true;
    }
    
    // Right boundary
    if (c.position.x + c.radius > arenaWidth) {
        c.position.x = arenaWidth - c.radius;
        c.velocity.x = -c.velocity.x * elasticity;
        collided = true;
    }
    
    // Bottom boundary
    if (c.position.y - c.radius < 0.0f) {
        c.position.y = c.radius;
        c.velocity.y = -c.velocity.y * elasticity;
        collided = true;
    }
    
    // Top boundary
    if (c.position.y + c.radius > arenaHeight) {
        c.position.y = arenaHeight - c.radius;
        c.velocity.y = -c.velocity.y * elasticity;
        collided = true;
    }
}

void PhysicsManager::updatePhysics(float deltaTime) {
    if (activeCount_ == 0) return;
    
    // Launch kernel with 256 threads per block (optimal for Ampere architecture)
    const int threadsPerBlock = 256;
    const int numBlocks = (activeCount_ + threadsPerBlock - 1) / threadsPerBlock;
    
    updatePhysicsKernel<<<numBlocks, threadsPerBlock, 0, physicsStream_>>>(
        d_circles_,
        activeCount_,
        deltaTime,
        arenaWidth_,
        arenaHeight_,
        elasticity_
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

void PhysicsManager::applyBoundaryConditions(float arenaWidth, float arenaHeight) {
    // Boundary conditions are now handled directly in updatePhysicsKernel
    // This method is kept for API compatibility but doesn't need to do anything
}

float PhysicsManager::calculateScaleFactor(float userZoomFactor) const {
    // Calculate automatic scale factor based on circle count
    // Scale inversely proportional to sqrt(count) when count > 10,000
    float autoScaleFactor = 1.0f;
    
    if (activeCount_ > 10000) {
        // Scale down as population increases
        // Formula: scale = sqrt(10000 / activeCount)
        autoScaleFactor = std::sqrt(10000.0f / static_cast<float>(activeCount_));
    }
    
    // Combine with user zoom factor (clamped to 10-500%)
    float clampedZoom = std::max(0.1f, std::min(5.0f, userZoomFactor));
    
    return autoScaleFactor * clampedZoom;
}

void PhysicsManager::cleanup() {
    if (d_circles_) {
        cudaFree(d_circles_);
        d_circles_ = nullptr;
    }
    
    if (physicsStream_) {
        cudaStreamDestroy(physicsStream_);
        physicsStream_ = nullptr;
    }
}
