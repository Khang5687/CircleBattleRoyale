#include "texture_atlas_manager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>

namespace fs = std::filesystem;

// Placeholder texture data (1x1 magenta pixel)
static const unsigned char PLACEHOLDER_TEXTURE[] = {255, 0, 255, 255};

struct ImageData {
    unsigned char* data;
    int width;
    int height;
    int channels;
    std::string filename;
    bool isPlaceholder;
    
    ImageData() : data(nullptr), width(0), height(0), channels(0), isPlaceholder(false) {}
    
    // Move constructor
    ImageData(ImageData&& other) noexcept 
        : data(other.data), width(other.width), height(other.height), 
          channels(other.channels), filename(std::move(other.filename)), 
          isPlaceholder(other.isPlaceholder) {
        other.data = nullptr;
    }
    
    // Move assignment
    ImageData& operator=(ImageData&& other) noexcept {
        if (this != &other) {
            if (data && !isPlaceholder) {
                stbi_image_free(data);
            }
            data = other.data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            filename = std::move(other.filename);
            isPlaceholder = other.isPlaceholder;
            other.data = nullptr;
        }
        return *this;
    }
    
    ~ImageData() {
        if (data && !isPlaceholder) {
            stbi_image_free(data);
        }
    }
    
    // Delete copy constructor and assignment
    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;
};

// Store loaded images as member variable
static std::vector<ImageData> g_loadedImages;

bool TextureAtlasManager::loadAvatars(const std::string& avatarDir) {
    // Clear any previously loaded images
    g_loadedImages.clear();
    
    // Check if directory exists
    if (!fs::exists(avatarDir) || !fs::is_directory(avatarDir)) {
        std::cerr << "Error: Avatar directory does not exist: " << avatarDir << std::endl;
        return false;
    }
    
    std::cout << "Loading avatars from: " << avatarDir << std::endl;
    
    // Iterate through all files in the directory
    int loadedCount = 0;
    int failedCount = 0;
    
    for (const auto& entry : fs::directory_iterator(avatarDir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filepath = entry.path().string();
        std::string extension = entry.path().extension().string();
        
        // Convert extension to lowercase
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        // Check if file is JPG or PNG
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png") {
            continue;
        }
        
        ImageData img;
        img.filename = entry.path().filename().string();
        
        // Load image using stb_image
        img.data = stbi_load(filepath.c_str(), &img.width, &img.height, &img.channels, 4); // Force RGBA
        
        if (img.data) {
            img.channels = 4; // We forced RGBA
            loadedCount++;
            if (loadedCount <= 10) { // Only print first 10 to avoid spam
                std::cout << "Loaded: " << img.filename << " (" << img.width << "x" << img.height << ")" << std::endl;
            }
        } else {
            // Failed to load - use placeholder
            std::cerr << "Warning: Failed to load " << filepath << " - using placeholder" << std::endl;
            img.data = const_cast<unsigned char*>(PLACEHOLDER_TEXTURE);
            img.width = 1;
            img.height = 1;
            img.channels = 4;
            img.isPlaceholder = true;
            failedCount++;
        }
        
        g_loadedImages.push_back(std::move(img));
    }
    
    std::cout << "Avatar loading complete: " << loadedCount << " loaded, " 
              << failedCount << " failed (using placeholders)" << std::endl;
    
    if (g_loadedImages.empty()) {
        std::cerr << "Error: No valid image files found in " << avatarDir << std::endl;
        return false;
    }
    
    return true;
}

bool TextureAtlasManager::createAtlas(uint32_t maxAtlasSize) {
    if (g_loadedImages.empty()) {
        std::cerr << "Error: No images loaded. Call loadAvatars() first." << std::endl;
        return false;
    }
    
    std::cout << "Creating texture atlas (max size: " << maxAtlasSize << "x" << maxAtlasSize << ")" << std::endl;
    
    // Clear previous atlas entries
    atlasEntries_.clear();
    atlasData_.clear();
    
    // Simple row-based packing algorithm
    uint32_t currentX = 0;
    uint32_t currentY = 0;
    uint32_t rowHeight = 0;
    uint32_t currentAtlasIndex = 0;
    
    // Allocate atlas buffer
    atlasData_.resize(maxAtlasSize * maxAtlasSize * 4, 0);
    atlasWidth_ = maxAtlasSize;
    atlasHeight_ = maxAtlasSize;
    
    for (size_t i = 0; i < g_loadedImages.size(); ++i) {
        const ImageData& img = g_loadedImages[i];
        
        // Check if we need to move to next row
        if (currentX + img.width > maxAtlasSize) {
            currentX = 0;
            currentY += rowHeight;
            rowHeight = 0;
        }
        
        // Check if we need a new atlas
        if (currentY + img.height > maxAtlasSize) {
            std::cout << "Warning: Need multiple atlases. Current implementation supports only one atlas." << std::endl;
            std::cout << "Packed " << i << " images into atlas " << currentAtlasIndex << std::endl;
            
            // For now, we'll just stop packing and use what we have
            // In a full implementation, we'd create a new atlas
            break;
        }
        
        // Copy image data to atlas
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int srcIdx = (y * img.width + x) * 4;
                int dstIdx = ((currentY + y) * maxAtlasSize + (currentX + x)) * 4;
                
                atlasData_[dstIdx + 0] = img.data[srcIdx + 0]; // R
                atlasData_[dstIdx + 1] = img.data[srcIdx + 1]; // G
                atlasData_[dstIdx + 2] = img.data[srcIdx + 2]; // B
                atlasData_[dstIdx + 3] = img.data[srcIdx + 3]; // A
            }
        }
        
        // Calculate UV coordinates (normalized to 0-1)
        float u0 = static_cast<float>(currentX) / maxAtlasSize;
        float v0 = static_cast<float>(currentY) / maxAtlasSize;
        float u1 = static_cast<float>(currentX + img.width) / maxAtlasSize;
        float v1 = static_cast<float>(currentY + img.height) / maxAtlasSize;
        
        // Create atlas entry
        AtlasEntry entry;
        entry.atlasIndex = currentAtlasIndex;
        entry.u0 = u0;
        entry.v0 = v0;
        entry.u1 = u1;
        entry.v1 = v1;
        entry.width = img.width;
        entry.height = img.height;
        
        atlasEntries_.push_back(entry);
        
        // Update position for next image
        currentX += img.width;
        rowHeight = std::max(rowHeight, static_cast<uint32_t>(img.height));
    }
    
    std::cout << "Atlas creation complete: " << atlasEntries_.size() << " images packed" << std::endl;
    std::cout << "Atlas utilization: " << currentY + rowHeight << " / " << maxAtlasSize 
              << " rows (" << (100.0f * (currentY + rowHeight) / maxAtlasSize) << "%)" << std::endl;
    
    return true;
}

// Helper function to generate mipmap level
static std::vector<unsigned char> generateMipmap(const unsigned char* srcData, 
                                                  uint32_t srcWidth, uint32_t srcHeight) {
    uint32_t dstWidth = srcWidth / 2;
    uint32_t dstHeight = srcHeight / 2;
    
    if (dstWidth == 0) dstWidth = 1;
    if (dstHeight == 0) dstHeight = 1;
    
    std::vector<unsigned char> dstData(dstWidth * dstHeight * 4);
    
    // Simple box filter (2x2 average)
    for (uint32_t y = 0; y < dstHeight; ++y) {
        for (uint32_t x = 0; x < dstWidth; ++x) {
            uint32_t srcX = x * 2;
            uint32_t srcY = y * 2;
            
            // Sample 4 pixels and average
            uint32_t r = 0, g = 0, b = 0, a = 0;
            int samples = 0;
            
            for (int dy = 0; dy < 2 && (srcY + dy) < srcHeight; ++dy) {
                for (int dx = 0; dx < 2 && (srcX + dx) < srcWidth; ++dx) {
                    uint32_t srcIdx = ((srcY + dy) * srcWidth + (srcX + dx)) * 4;
                    r += srcData[srcIdx + 0];
                    g += srcData[srcIdx + 1];
                    b += srcData[srcIdx + 2];
                    a += srcData[srcIdx + 3];
                    samples++;
                }
            }
            
            uint32_t dstIdx = (y * dstWidth + x) * 4;
            dstData[dstIdx + 0] = r / samples;
            dstData[dstIdx + 1] = g / samples;
            dstData[dstIdx + 2] = b / samples;
            dstData[dstIdx + 3] = a / samples;
        }
    }
    
    return dstData;
}

bool TextureAtlasManager::uploadToGPU() {
    if (atlasData_.empty()) {
        std::cerr << "Error: No atlas data. Call createAtlas() first." << std::endl;
        return false;
    }
    
    std::cout << "Uploading texture atlas to GPU with mipmaps..." << std::endl;
    
    // Generate 4 mipmap levels as specified
    const int numMipmapLevels = 4;
    std::vector<std::vector<unsigned char>> mipmaps;
    mipmaps.push_back(atlasData_); // Level 0 (full resolution)
    
    uint32_t mipWidth = atlasWidth_;
    uint32_t mipHeight = atlasHeight_;
    
    for (int level = 1; level < numMipmapLevels; ++level) {
        std::vector<unsigned char> mipData = generateMipmap(
            mipmaps[level - 1].data(), mipWidth, mipHeight);
        
        mipWidth = mipWidth / 2;
        mipHeight = mipHeight / 2;
        if (mipWidth == 0) mipWidth = 1;
        if (mipHeight == 0) mipHeight = 1;
        
        mipmaps.push_back(std::move(mipData));
        
        std::cout << "  Generated mipmap level " << level << ": " 
                  << mipWidth << "x" << mipHeight << std::endl;
    }
    
    // Create CUDA mipmapped array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaExtent extent = make_cudaExtent(atlasWidth_, atlasHeight_, 0);
    
    cudaError_t err = cudaMallocMipmappedArray(&cudaMipmappedArray_, &channelDesc, 
                                                extent, numMipmapLevels);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to allocate CUDA mipmapped array: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Upload each mipmap level
    mipWidth = atlasWidth_;
    mipHeight = atlasHeight_;
    
    for (int level = 0; level < numMipmapLevels; ++level) {
        cudaArray_t levelArray;
        err = cudaGetMipmappedArrayLevel(&levelArray, cudaMipmappedArray_, level);
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to get mipmap level " << level << ": " 
                      << cudaGetErrorString(err) << std::endl;
            cudaFreeMipmappedArray(cudaMipmappedArray_);
            return false;
        }
        
        err = cudaMemcpy2DToArray(levelArray, 0, 0, mipmaps[level].data(),
                                   mipWidth * 4, mipWidth * 4, mipHeight,
                                   cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to upload mipmap level " << level << ": " 
                      << cudaGetErrorString(err) << std::endl;
            cudaFreeMipmappedArray(cudaMipmappedArray_);
            return false;
        }
        
        mipWidth = mipWidth / 2;
        mipHeight = mipHeight / 2;
        if (mipWidth == 0) mipWidth = 1;
        if (mipHeight == 0) mipHeight = 1;
    }
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = cudaMipmappedArray_;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 1;
    texDesc.maxMipmapLevelClamp = numMipmapLevels - 1;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    
    err = cudaCreateTextureObject(&cudaTexture_, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to create texture object: " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFreeMipmappedArray(cudaMipmappedArray_);
        return false;
    }
    
    std::cout << "Texture atlas uploaded successfully to GPU" << std::endl;
    std::cout << "  Base resolution: " << atlasWidth_ << "x" << atlasHeight_ << std::endl;
    std::cout << "  Mipmap levels: " << numMipmapLevels << std::endl;
    std::cout << "  Memory usage (uncompressed): " 
              << (atlasData_.size() / (1024.0f * 1024.0f)) << " MB" << std::endl;
    
    return true;
}

const TextureAtlasManager::AtlasEntry& TextureAtlasManager::getAtlasEntry(uint32_t circleID) const {
    static AtlasEntry dummy = {0, 0.0f, 0.0f, 1.0f, 1.0f, 0, 0};
    
    if (circleID < atlasEntries_.size()) {
        return atlasEntries_[circleID];
    }
    
    return dummy;
}

// Constructor to initialize cache
TextureAtlasManager::TextureAtlasManager() 
    : cudaArray_(nullptr), cudaMipmappedArray_(nullptr), cudaTexture_(0),
      glTextureID_(0), atlasWidth_(0), atlasHeight_(0),
      cacheSize_(MAX_RESIDENT_TEXTURES), currentTime_(0) {
}

// Destructor to clean up GPU resources
TextureAtlasManager::~TextureAtlasManager() {
    if (cudaTexture_) {
        cudaDestroyTextureObject(cudaTexture_);
    }
    if (cudaMipmappedArray_) {
        cudaFreeMipmappedArray(cudaMipmappedArray_);
    }
}


// Streaming texture management implementation

void TextureAtlasManager::updateVisibleTextures(const std::vector<uint32_t>& visibleCircleIDs) {
    currentTime_++;
    
    // Mark visible textures as accessed
    for (uint32_t circleID : visibleCircleIDs) {
        if (circleID >= atlasEntries_.size()) continue;
        
        // Update access time
        residentTextures_[circleID] = currentTime_;
        
        // Move to front of LRU list
        lruList_.remove(circleID);
        lruList_.push_front(circleID);
        
        // Check if we need to evict textures
        while (residentTextures_.size() > MAX_RESIDENT_TEXTURES) {
            evictLRUTexture();
        }
    }
}

void TextureAtlasManager::evictLRUTexture() {
    if (lruList_.empty()) return;
    
    // Get least recently used texture
    uint32_t lruCircleID = lruList_.back();
    lruList_.pop_back();
    
    // Remove from resident set
    residentTextures_.erase(lruCircleID);
    
    // In a full implementation, we would:
    // 1. Free GPU memory for this texture
    // 2. Update atlas to remove this texture
    // For now, we just track it in the cache
}

void TextureAtlasManager::loadTexture(uint32_t circleID) {
    if (circleID >= atlasEntries_.size()) return;
    
    // In a full implementation, we would:
    // 1. Load texture data from disk or CPU memory
    // 2. Upload to GPU
    // 3. Update atlas entry
    
    // For now, we just mark it as resident
    residentTextures_[circleID] = currentTime_;
    lruList_.push_front(circleID);
    
    // Evict if necessary
    while (residentTextures_.size() > MAX_RESIDENT_TEXTURES) {
        evictLRUTexture();
    }
}

bool TextureAtlasManager::isTextureResident(uint32_t circleID) const {
    return residentTextures_.find(circleID) != residentTextures_.end();
}
