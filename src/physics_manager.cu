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

// CUDA kernel to mark dead circles and compact array
__global__ void markDeadCirclesKernel(
    const PhysicsManager::Circle* circles,
    uint32_t count,
    uint32_t* aliveFlags
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Mark as alive (1) if health > 0, dead (0) otherwise
    aliveFlags[idx] = (circles[idx].health > 0.0f) ? 1 : 0;
}

// CUDA kernel for parallel prefix sum (scan) - simplified version
__global__ void prefixSumKernel(
    const uint32_t* input,
    uint32_t* output,
    uint32_t count
) {
    extern __shared__ uint32_t temp[];
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (idx < count) ? input[idx] : 0;
    __syncthreads();
    
    // Up-sweep phase
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            uint32_t t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }
    
    // Write output
    if (idx < count) {
        output[idx] = temp[tid] + input[idx];
    }
}

// CUDA kernel to compact alive circles
__global__ void compactCirclesKernel(
    const PhysicsManager::Circle* inputCircles,
    PhysicsManager::Circle* outputCircles,
    const uint32_t* aliveFlags,
    const uint32_t* prefixSum,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // If this circle is alive, copy it to its new position
    if (aliveFlags[idx] == 1) {
        uint32_t newIdx = prefixSum[idx] - 1;  // Prefix sum gives us the new index
        outputCircles[newIdx] = inputCircles[idx];
    }
}

void PhysicsManager::removeDeadCircles() {
    if (activeCount_ == 0) return;
    
    // Allocate temporary buffers
    uint32_t* d_aliveFlags;
    uint32_t* d_prefixSum;
    PhysicsManager::Circle* d_tempCircles;
    
    cudaMalloc(&d_aliveFlags, activeCount_ * sizeof(uint32_t));
    cudaMalloc(&d_prefixSum, activeCount_ * sizeof(uint32_t));
    cudaMalloc(&d_tempCircles, activeCount_ * sizeof(Circle));
    
    // Mark dead circles
    const int threadsPerBlock = 256;
    const int numBlocks = (activeCount_ + threadsPerBlock - 1) / threadsPerBlock;
    
    markDeadCirclesKernel<<<numBlocks, threadsPerBlock, 0, physicsStream_>>>(
        d_circles_,
        activeCount_,
        d_aliveFlags
    );
    
    // Compute prefix sum for compaction
    // For simplicity, use a CPU-based approach for now (can be optimized with thrust or custom GPU scan)
    uint32_t* h_aliveFlags = new uint32_t[activeCount_];
    uint32_t* h_prefixSum = new uint32_t[activeCount_];
    
    cudaMemcpy(h_aliveFlags, d_aliveFlags, activeCount_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // CPU prefix sum
    uint32_t sum = 0;
    for (uint32_t i = 0; i < activeCount_; i++) {
        sum += h_aliveFlags[i];
        h_prefixSum[i] = sum;
    }
    
    uint32_t newActiveCount = sum;
    
    // Copy prefix sum back to device
    cudaMemcpy(d_prefixSum, h_prefixSum, activeCount_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Compact circles
    compactCirclesKernel<<<numBlocks, threadsPerBlock, 0, physicsStream_>>>(
        d_circles_,
        d_tempCircles,
        d_aliveFlags,
        d_prefixSum,
        activeCount_
    );
    
    // Copy compacted array back
    if (newActiveCount > 0) {
        cudaMemcpy(d_circles_, d_tempCircles, newActiveCount * sizeof(Circle), cudaMemcpyDeviceToDevice);
    }
    
    // Update active count
    uint32_t removedCount = activeCount_ - newActiveCount;
    if (removedCount > 0) {
        std::cout << "Removed " << removedCount << " dead circles. Active: " << newActiveCount << std::endl;
    }
    
    activeCount_ = newActiveCount;
    
    // Cleanup
    delete[] h_aliveFlags;
    delete[] h_prefixSum;
    cudaFree(d_aliveFlags);
    cudaFree(d_prefixSum);
    cudaFree(d_tempCircles);
    
    // Check for winner
    if (activeCount_ == 1) {
        std::cout << "WINNER DETECTED! Only 1 circle remains!" << std::endl;
    }
}

uint32_t PhysicsManager::getWinnerID() const {
    if (activeCount_ != 1) {
        return UINT32_MAX;  // No winner yet
    }
    
    // Copy the last circle to host to get its avatar ID
    Circle h_winner;
    cudaMemcpy(&h_winner, d_circles_, sizeof(Circle), cudaMemcpyDeviceToHost);
    
    return h_winner.avatarID;
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
