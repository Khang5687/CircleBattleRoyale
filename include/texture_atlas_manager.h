#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

class TextureAtlasManager {
public:
    struct AtlasEntry {
        uint32_t atlasIndex;
        float u0, v0, u1, v1;  // UV coordinates
        uint32_t width, height;
    };
    
    bool loadAvatars(const std::string& avatarDir);
    bool createAtlas(uint32_t maxAtlasSize = 8192);
    bool uploadToGPU();
    
    cudaTextureObject_t getCudaTextureObject() const { return cudaTexture_; }
    unsigned int getOpenGLTextureID() const { return glTextureID_; }
    const AtlasEntry& getAtlasEntry(uint32_t circleID) const;
    
private:
    std::vector<AtlasEntry> atlasEntries_;
    cudaArray_t cudaArray_;
    cudaTextureObject_t cudaTexture_;
    unsigned int glTextureID_;
};
