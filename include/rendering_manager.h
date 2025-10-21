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
    unsigned int vao_;
    unsigned int instanceVBO_;
    unsigned int shaderProgram_;
    cudaGraphicsResource_t cudaVBOResource_;
    
    void setupShaders();
    void setupGeometry();
    uint32_t selectLODLevel(float screenRadius);
};
