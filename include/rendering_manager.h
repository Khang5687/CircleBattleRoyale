#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "physics_manager.h"
#include "texture_atlas_manager.h"

struct GLFWwindow;

class RenderingManager {
public:
    bool initialize(uint32_t maxCircles, GLFWwindow* window);
    void render(const PhysicsManager::Circle* circles, uint32_t count, 
                float zoomFactor, const TextureAtlasManager& atlasManager);
    
    void cleanup();
    
private:
    // OpenGL resources
    unsigned int vao_;
    unsigned int quadVBO_;
    unsigned int instanceVBO_;
    unsigned int shaderProgram_;
    
    // CUDA-OpenGL interop
    cudaGraphicsResource_t cudaVBOResource_;
    cudaStream_t renderStream_;
    
    // Frustum culling
    uint32_t* d_visibleIndices_;
    uint32_t* d_visibleCount_;
    uint32_t* h_visibleCount_;
    
    // Window reference
    GLFWwindow* window_;
    uint32_t maxCircles_;
    
    // Projection matrix
    float projectionMatrix_[16];
    
    void setupShaders();
    void setupGeometry();
    bool setupCUDAInterop();
    void updateInstanceData(const PhysicsManager::Circle* circles, uint32_t count, float zoomFactor);
    uint32_t selectLODLevel(float screenRadius);
};
