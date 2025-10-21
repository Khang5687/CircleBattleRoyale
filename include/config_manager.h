#pragma once

#include <string>
#include <vector>
#include <cstdint>

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
    
    const SimConfig& getSimConfig() const { return simConfig_; }
    const std::vector<BiasEntry>& getBiasEntries() const { return biasEntries_; }
    const std::vector<DamageScalingEntry>& getDamageScaling() const { return damageScaling_; }

private:
    SimConfig simConfig_;
    std::vector<BiasEntry> biasEntries_;
    std::vector<DamageScalingEntry> damageScaling_;
};
