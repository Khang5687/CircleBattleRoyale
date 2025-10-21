#include <iostream>
#include <memory>
#include <GLFW/glfw3.h>
#include "config_manager.h"

int main(int argc, char** argv) {
    std::cout << "CUDA Battle Royale Simulation" << std::endl;
    std::cout << "Initializing..." << std::endl;
    
    // Load configuration
    ConfigManager configManager;
    if (!configManager.loadConfig("config/config.txt")) {
        std::cerr << "Failed to load configuration" << std::endl;
        return -1;
    }
    
    // Load bias and damage scaling (optional)
    configManager.loadBias("config/bias.txt");
    configManager.loadDamageScaling("config/damage_scaling.txt");
    
    // Enable hot-reload for bias and damage scaling
    configManager.enableHotReload("config/bias.txt", "config/damage_scaling.txt");
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    std::cout << "GLFW initialized successfully" << std::endl;
    
    // Main loop placeholder - hot-reload will be checked here
    // In the actual simulation loop, call: configManager.checkForUpdates();
    
    // Cleanup
    glfwTerminate();
    
    return 0;
}
