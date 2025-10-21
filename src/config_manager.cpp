#include "config_manager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

// Helper function to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Helper function to parse key-value pairs
static bool parseKeyValue(const std::string& line, std::string& key, std::string& value) {
    size_t pos = line.find('=');
    if (pos == std::string::npos) return false;
    
    key = trim(line.substr(0, pos));
    value = trim(line.substr(pos + 1));
    return !key.empty() && !value.empty();
}

bool ConfigManager::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file: " << configPath << std::endl;
        return false;
    }
    
    // Initialize with default values
    simConfig_.initialCircleCount = 10000;
    simConfig_.arenaWidth = 10000.0f;
    simConfig_.arenaHeight = 10000.0f;
    simConfig_.baseCircleRadius = 50.0f;
    simConfig_.baseCircleMass = 1.0f;
    simConfig_.elasticity = 0.95f;
    simConfig_.baseHealth = 100.0f;
    
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::string key, value;
        if (!parseKeyValue(line, key, value)) {
            std::cerr << "Warning: Invalid line " << lineNumber << " in " << configPath << ": " << line << std::endl;
            continue;
        }
        
        try {
            if (key == "initialCircleCount") {
                simConfig_.initialCircleCount = std::stoul(value);
            } else if (key == "arenaWidth") {
                simConfig_.arenaWidth = std::stof(value);
            } else if (key == "arenaHeight") {
                simConfig_.arenaHeight = std::stof(value);
            } else if (key == "baseCircleRadius") {
                simConfig_.baseCircleRadius = std::stof(value);
            } else if (key == "baseCircleMass") {
                simConfig_.baseCircleMass = std::stof(value);
            } else if (key == "elasticity") {
                simConfig_.elasticity = std::stof(value);
            } else if (key == "baseHealth") {
                simConfig_.baseHealth = std::stof(value);
            } else {
                std::cerr << "Warning: Unknown config key '" << key << "' at line " << lineNumber << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Failed to parse value for '" << key << "' at line " << lineNumber << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    // Validate parameters
    if (simConfig_.initialCircleCount < 1 || simConfig_.initialCircleCount > 1000000) {
        std::cerr << "Error: initialCircleCount must be between 1 and 1,000,000 (got " << simConfig_.initialCircleCount << ")" << std::endl;
        return false;
    }
    
    if (simConfig_.arenaWidth < 1000.0f || simConfig_.arenaHeight < 1000.0f) {
        std::cerr << "Error: Arena dimensions must be at least 1000x1000 pixels (got " 
                  << simConfig_.arenaWidth << "x" << simConfig_.arenaHeight << ")" << std::endl;
        return false;
    }
    
    if (simConfig_.baseCircleRadius <= 0.0f) {
        std::cerr << "Error: baseCircleRadius must be positive (got " << simConfig_.baseCircleRadius << ")" << std::endl;
        return false;
    }
    
    if (simConfig_.baseCircleMass <= 0.0f) {
        std::cerr << "Error: baseCircleMass must be positive (got " << simConfig_.baseCircleMass << ")" << std::endl;
        return false;
    }
    
    if (simConfig_.elasticity < 0.0f || simConfig_.elasticity > 1.0f) {
        std::cerr << "Error: elasticity must be between 0.0 and 1.0 (got " << simConfig_.elasticity << ")" << std::endl;
        return false;
    }
    
    if (simConfig_.baseHealth <= 0.0f) {
        std::cerr << "Error: baseHealth must be positive (got " << simConfig_.baseHealth << ")" << std::endl;
        return false;
    }
    
    std::cout << "Configuration loaded successfully:" << std::endl;
    std::cout << "  Circle Count: " << simConfig_.initialCircleCount << std::endl;
    std::cout << "  Arena Size: " << simConfig_.arenaWidth << "x" << simConfig_.arenaHeight << std::endl;
    std::cout << "  Base Radius: " << simConfig_.baseCircleRadius << std::endl;
    std::cout << "  Base Mass: " << simConfig_.baseCircleMass << std::endl;
    std::cout << "  Elasticity: " << simConfig_.elasticity << std::endl;
    std::cout << "  Base Health: " << simConfig_.baseHealth << std::endl;
    
    return true;
}

bool ConfigManager::loadBias(const std::string& biasPath) {
    std::ifstream file(biasPath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open bias file: " << biasPath << " (using no bias)" << std::endl;
        biasEntries_.clear();
        return true;  // Not having a bias file is acceptable
    }
    
    biasEntries_.clear();
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        BiasEntry entry;
        
        if (!(iss >> entry.playerName >> entry.biasMultiplier)) {
            std::cerr << "Warning: Invalid bias entry at line " << lineNumber << ": " << line << std::endl;
            continue;
        }
        
        // Validate bias multiplier
        if (entry.biasMultiplier <= 0.0f) {
            std::cerr << "Warning: Bias multiplier must be positive at line " << lineNumber 
                      << " (got " << entry.biasMultiplier << "), skipping" << std::endl;
            continue;
        }
        
        if (entry.biasMultiplier > 100.0f) {
            std::cerr << "Warning: Bias multiplier seems unreasonably high at line " << lineNumber 
                      << " (got " << entry.biasMultiplier << "), capping at 100.0" << std::endl;
            entry.biasMultiplier = 100.0f;
        }
        
        biasEntries_.push_back(entry);
    }
    
    std::cout << "Loaded " << biasEntries_.size() << " bias entries" << std::endl;
    return true;
}

bool ConfigManager::loadDamageScaling(const std::string& scalingPath) {
    std::ifstream file(scalingPath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open damage scaling file: " << scalingPath << " (using no scaling)" << std::endl;
        damageScaling_.clear();
        return true;  // Not having a damage scaling file is acceptable
    }
    
    damageScaling_.clear();
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        DamageScalingEntry entry;
        
        if (!(iss >> entry.populationThreshold >> entry.damageMultiplier)) {
            std::cerr << "Warning: Invalid damage scaling entry at line " << lineNumber << ": " << line << std::endl;
            continue;
        }
        
        // Validate damage multiplier
        if (entry.damageMultiplier <= 0.0f) {
            std::cerr << "Warning: Damage multiplier must be positive at line " << lineNumber 
                      << " (got " << entry.damageMultiplier << "), skipping" << std::endl;
            continue;
        }
        
        if (entry.damageMultiplier > 1000.0f) {
            std::cerr << "Warning: Damage multiplier seems unreasonably high at line " << lineNumber 
                      << " (got " << entry.damageMultiplier << "), capping at 1000.0" << std::endl;
            entry.damageMultiplier = 1000.0f;
        }
        
        damageScaling_.push_back(entry);
    }
    
    // Sort by population threshold (descending order for easier lookup)
    std::sort(damageScaling_.begin(), damageScaling_.end(), 
              [](const DamageScalingEntry& a, const DamageScalingEntry& b) {
                  return a.populationThreshold > b.populationThreshold;
              });
    
    std::cout << "Loaded " << damageScaling_.size() << " damage scaling entries" << std::endl;
    return true;
}

void ConfigManager::enableHotReload(const std::string& biasPath, const std::string& scalingPath) {
    hotReloadEnabled_ = true;
    biasPath_ = biasPath;
    scalingPath_ = scalingPath;
    
    // Initialize last modified times
    try {
        if (std::filesystem::exists(biasPath_)) {
            biasLastModified_ = std::filesystem::last_write_time(biasPath_);
        }
        if (std::filesystem::exists(scalingPath_)) {
            scalingLastModified_ = std::filesystem::last_write_time(scalingPath_);
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to initialize hot-reload: " << e.what() << std::endl;
        hotReloadEnabled_ = false;
    }
    
    if (hotReloadEnabled_) {
        std::cout << "Hot-reload enabled for bias and damage scaling files" << std::endl;
    }
}

void ConfigManager::checkForUpdates() {
    if (!hotReloadEnabled_) return;
    
    try {
        // Check bias file
        if (std::filesystem::exists(biasPath_)) {
            auto currentModTime = std::filesystem::last_write_time(biasPath_);
            if (currentModTime != biasLastModified_) {
                std::cout << "Detected change in bias file, reloading..." << std::endl;
                if (loadBias(biasPath_)) {
                    biasLastModified_ = currentModTime;
                    std::cout << "Bias configuration reloaded successfully" << std::endl;
                } else {
                    std::cerr << "Failed to reload bias configuration" << std::endl;
                }
            }
        }
        
        // Check damage scaling file
        if (std::filesystem::exists(scalingPath_)) {
            auto currentModTime = std::filesystem::last_write_time(scalingPath_);
            if (currentModTime != scalingLastModified_) {
                std::cout << "Detected change in damage scaling file, reloading..." << std::endl;
                if (loadDamageScaling(scalingPath_)) {
                    scalingLastModified_ = currentModTime;
                    std::cout << "Damage scaling configuration reloaded successfully" << std::endl;
                } else {
                    std::cerr << "Failed to reload damage scaling configuration" << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error checking for config updates: " << e.what() << std::endl;
    }
}
