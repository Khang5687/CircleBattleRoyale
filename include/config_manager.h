#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <filesystem>

class ConfigManager {
public:
    struct SimConfig {
        uint32_t initialCircleCount;
        float arenaWidth;
        float arenaHeight;
        float baseCircleRadius;
        float baseCircleMass;
        float elasticity;
        float baseHealth;
    };
    
    struct BiasEntry {
        std::string playerName;
        float biasMultiplier;
    };
    
    struct DamageScalingEntry {
        uint32_t populationThreshold;
        float damageMultiplier;
    };
    
    bool loadConfig(const std::string& configPath);
    bool loadBias(const std::string& biasPath);
    bool loadDamageScaling(const std::string& scalingPath);
    
    // Hot-reload functionality
    void enableHotReload(const std::string& biasPath, const std::string& scalingPath);
    void checkForUpdates();  // Call this in the main loop
    
    const SimConfig& getSimConfig() const { return simConfig_; }
    const std::vector<BiasEntry>& getBiasEntries() const { return biasEntries_; }
    const std::vector<DamageScalingEntry>& getDamageScaling() const { return damageScaling_; }

private:
    SimConfig simConfig_;
    std::vector<BiasEntry> biasEntries_;
    std::vector<DamageScalingEntry> damageScaling_;
    
    // Hot-reload state
    bool hotReloadEnabled_ = false;
    std::string biasPath_;
    std::string scalingPath_;
    std::filesystem::file_time_type biasLastModified_;
    std::filesystem::file_time_type scalingLastModified_;
};
