#include <cuda_runtime.h>
#include "physics_manager.h"

// Instance data structure for rendering
struct InstanceData {
    float2 position;
    float radius;
    float lod;
    float4 uvs;  // u0, v0, u1, v1
    float health;
    float padding[2];
};

// Frustum planes (left, right, bottom, top)
struct Frustum {
    float left;
    float right;
    float bottom;
    float top;
};

__device__ uint32_t selectLODLevel(float screenRadius) {
    // LOD levels based on screen-space size
    if (screenRadius > 64.0f) return 0;  // Full resolution
    if (screenRadius > 32.0f) return 1;  // 1/2 resolution
    if (screenRadius > 16.0f) return 2;  // 1/4 resolution
    return 3;  // 1/8 resolution
}

__device__ bool isCircleVisible(const PhysicsManager::Circle& circle, const Frustum& frustum, float zoomFactor) {
    float effectiveRadius = circle.radius * zoomFactor;
    
    // Check if circle is outside frustum
    if (circle.position.x + effectiveRadius < frustum.left) return false;
    if (circle.position.x - effectiveRadius > frustum.right) return false;
    if (circle.position.y + effectiveRadius < frustum.bottom) return false;
    if (circle.position.y - effectiveRadius > frustum.top) return false;
    
    return true;
}

__global__ void populateInstanceDataKernel(
    const PhysicsManager::Circle* circles,
    uint32_t count,
    InstanceData* instanceData,
    uint32_t* visibleIndices,
    uint32_t* visibleCount,
    float zoomFactor,
    Frustum frustum
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    const PhysicsManager::Circle& c = circles[idx];
    
    // Frustum culling
    if (!isCircleVisible(c, frustum, zoomFactor)) {
        return;
    }
    
    // Allocate slot in visible list
    uint32_t slot = atomicAdd(visibleCount, 1);
    visibleIndices[slot] = idx;
    
    // Calculate screen-space radius for LOD selection
    float screenRadius = c.radius * zoomFactor;
    uint32_t lodLevel = selectLODLevel(screenRadius);
    
    // Populate instance data
    InstanceData& inst = instanceData[slot];
    inst.position = c.position;
    inst.radius = c.radius;
    inst.lod = static_cast<float>(lodLevel);
    
    // TODO: Get UV coordinates from texture atlas
    // For now, use placeholder UVs
    inst.uvs = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
    
    inst.health = c.health;
    inst.padding[0] = 0.0f;
    inst.padding[1] = 0.0f;
}

// Host function to launch kernel
extern "C" void launchPopulateInstanceDataKernel(
    const PhysicsManager::Circle* d_circles,
    uint32_t count,
    void* d_instanceData,
    uint32_t* d_visibleIndices,
    uint32_t* d_visibleCount,
    float zoomFactor,
    float frustumLeft,
    float frustumRight,
    float frustumBottom,
    float frustumTop,
    cudaStream_t stream
) {
    Frustum frustum;
    frustum.left = frustumLeft;
    frustum.right = frustumRight;
    frustum.bottom = frustumBottom;
    frustum.top = frustumTop;
    
    // Reset visible count
    cudaMemsetAsync(d_visibleCount, 0, sizeof(uint32_t), stream);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    
    populateInstanceDataKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_circles,
        count,
        reinterpret_cast<InstanceData*>(d_instanceData),
        d_visibleIndices,
        d_visibleCount,
        zoomFactor,
        frustum
    );
}
