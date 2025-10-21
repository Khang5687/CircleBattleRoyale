#include <iostream>
#include <memory>
#include <GLFW/glfw3.h>

int main(int argc, char** argv) {
    std::cout << "CUDA Battle Royale Simulation" << std::endl;
    std::cout << "Initializing..." << std::endl;
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    std::cout << "GLFW initialized successfully" << std::endl;
    
    // Cleanup
    glfwTerminate();
    
    return 0;
}
