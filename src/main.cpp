#include <iostream>
#include <memory>
#include <GLFW/glfw3.h>
#include "config_manager.h"
#include "physics_manager.h"
#include "collision_manager.h"

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
    
    // Initialize physics manager
    PhysicsManager physicsManager;
    if (!physicsManager.initialize(1000000, configManager.getSimConfig())) {
        std::cerr << "Failed to initialize physics manager" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Apply bias multipliers
    physicsManager.applyBiasMultipliers(configManager.getBiasEntries());
    
    // Initialize collision manager
    // Cell size = 2 × max circle radius (optimal for spatial hashing)
    float cellSize = 2.0f * configManager.getSimConfig().baseCircleRadius;
    CollisionManager collisionManager;
    if (!collisionManager.initialize(
        1000000,
        configManager.getSimConfig().arenaWidth,
        configManager.getSimConfig().arenaHeight,
        cellSize
    )) {
        std::cerr << "Failed to initialize collision manager" << std::endl;
        physicsManager.cleanup();
        glfwTerminate();
        return -1;
    }
    
    std::cout << "\nSimulation initialized successfully!" << std::endl;
    std::cout << "Starting simulation loop..." << std::endl;
    
    // Main simulation loop
    double lastTime = glfwGetTime();
    int frameCount = 0;
    double fpsTimer = 0.0;
    int totalFrames = 0;
    const int maxFrames = 6000;  // Run for ~100 seconds at 60 FPS (for testing)
    
    while (totalFrames < maxFrames && !physicsManager.hasWinner()) {
        // Calculate delta time
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;
        
        // Cap delta time to prevent instability
        if (deltaTime > 0.1f) deltaTime = 0.1f;
        
        // Check for configuration updates
        configManager.checkForUpdates();
        
        // Get current damage multiplier based on population
        float damageMultiplier = 1.0f;
        uint32_t activeCount = physicsManager.getActiveCircleCount();
        const auto& damageScaling = configManager.getDamageScaling();
        
        for (const auto& entry : damageScaling) {
            if (activeCount <= entry.populationThreshold) {
                damageMultiplier = entry.damageMultiplier;
                break;
            }
        }
        
        // Update physics
        physicsManager.updatePhysics(deltaTime);
        
        // Detect and resolve collisions
        collisionManager.detectAndResolveCollisions(
            physicsManager.getCircleData(),
            activeCount,
            damageMultiplier
        );
        
        // Remove dead circles (check every 60 frames to reduce overhead)
        if (frameCount % 60 == 0) {
            physicsManager.removeDeadCircles();
        }
        
        // FPS counter
        frameCount++;
        totalFrames++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0) {
            std::cout << "FPS: " << frameCount << " | Active circles: " << physicsManager.getActiveCircleCount() << std::endl;
            frameCount = 0;
            fpsTimer = 0.0;
        }
        
        // Poll events
        glfwPollEvents();
        
        // Exit if ESC is pressed (for testing without window)
        // In full implementation, this would be handled by window callbacks
    }
    
    // Check for winner
    if (physicsManager.hasWinner()) {
        uint32_t winnerID = physicsManager.getWinnerID();
        std::cout << "\n========================================" << std::endl;
        std::cout << "SIMULATION COMPLETE!" << std::endl;
        std::cout << "Winner: Circle ID " << winnerID << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    collisionManager.cleanup();
    physicsManager.cleanup();
    glfwTerminate();
    
    std::cout << "Simulation ended successfully" << std::endl;
    
    return 0;
}
