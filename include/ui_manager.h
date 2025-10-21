#pragma once

#include <string>
#include <cstdint>

struct GLFWwindow;
struct ImGuiContext;

class UIManager {
public:
    struct PerformanceMetrics {
        float fps;
        uint32_t activeCircles;
        float cudaKernelTime;
        size_t gpuMemoryUsed;
        size_t systemMemoryUsed;
    };
    
    bool initialize(GLFWwindow* window);
    void render(const PerformanceMetrics& metrics, bool visible);
    void renderWinnerScreen(uint32_t winnerID, const std::string& winnerName);
    
    void cleanup();
    
private:
    ImGuiContext* imguiContext_;
    bool metricsVisible_;
};
