#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <list>
#include <cuda_runtime.h>

class TextureAtlasManager {
public:
    struct AtlasEntry {
        uint32_t atlasIndex;
        float u0, v0, u1, v1;  // UV coordinates
        uint32_t width, height;
    };
    
    TextureAtlasManager();
    ~TextureAtlasManager();
    
    bool loadAvatars(const std::string& avatarDir);
    bool createAtlas(uint32_t maxAtlasSize = 8192);
    bool uploadToGPU();
    
    // Streaming texture management
    void updateVisibleTextures(const std::vector<uint32_t>& visibleCircleIDs);
    bool isTextureResident(uint32_t circleID) const;
    size_t getCacheSize() const { return cacheSize_; }
    size_t getResidentTextureCount() const { return residentTextures_.size(); }
    
    cudaTextureObject_t getCudaTextureObject() const { return cudaTexture_; }
    unsigned int getOpenGLTextureID() const { return glTextureID_; }
    const AtlasEntry& getAtlasEntry(uint32_t circleID) const;
    
    uint32_t getAtlasWidth() const { return atlasWidth_; }
    uint32_t getAtlasHeight() const { return atlasHeight_; }
    
private:
    // LRU cache implementation
    struct CacheEntry {
        uint32_t circleID;
        uint64_t lastAccessTime;
    };
    
    void evictLRUTexture();
    void loadTexture(uint32_t circleID);
    
    std::vector<AtlasEntry> atlasEntries_;
    std::vector<unsigned char> atlasData_;  // CPU-side atlas data
    cudaArray_t cudaArray_;
    cudaMipmappedArray_t cudaMipmappedArray_;
    cudaTextureObject_t cudaTexture_;
    unsigned int glTextureID_;
    uint32_t atlasWidth_;
    uint32_t atlasHeight_;
    
    // Streaming cache
    static const size_t MAX_RESIDENT_TEXTURES = 2000;
    size_t cacheSize_;
    std::unordered_map<uint32_t, uint64_t> residentTextures_;  // circleID -> lastAccessTime
    std::list<uint32_t> lruList_;  // LRU order (front = most recent)
    uint64_t currentTime_;
};
