#include "texture_atlas_manager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

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
    
    ~ImageData() {
        if (data && !isPlaceholder) {
            stbi_image_free(data);
        }
    }
};

bool TextureAtlasManager::loadAvatars(const std::string& avatarDir) {
    std::vector<ImageData> images;
    
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
            std::cout << "Loaded: " << img.filename << " (" << img.width << "x" << img.height << ")" << std::endl;
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
        
        images.push_back(std::move(img));
    }
    
    std::cout << "Avatar loading complete: " << loadedCount << " loaded, " 
              << failedCount << " failed (using placeholders)" << std::endl;
    
    if (images.empty()) {
        std::cerr << "Error: No valid image files found in " << avatarDir << std::endl;
        return false;
    }
    
    // Store loaded images for atlas creation (will be implemented in task 3.2)
    // For now, just report success
    return true;
}

bool TextureAtlasManager::createAtlas(uint32_t maxAtlasSize) {
    // Will be implemented in task 3.2
    std::cout << "createAtlas() - Not yet implemented (task 3.2)" << std::endl;
    return false;
}

bool TextureAtlasManager::uploadToGPU() {
    // Will be implemented in task 3.3
    std::cout << "uploadToGPU() - Not yet implemented (task 3.3)" << std::endl;
    return false;
}

const TextureAtlasManager::AtlasEntry& TextureAtlasManager::getAtlasEntry(uint32_t circleID) const {
    static AtlasEntry dummy = {0, 0.0f, 0.0f, 1.0f, 1.0f, 0, 0};
    
    if (circleID < atlasEntries_.size()) {
        return atlasEntries_[circleID];
    }
    
    return dummy;
}
