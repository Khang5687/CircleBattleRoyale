#include "rendering_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Include OpenGL before GLFW
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

// OpenGL types (if not defined)
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif

// OpenGL function loader - we'll use GLFW's built-in loader
#ifndef APIENTRY
#define APIENTRY
#endif

// OpenGL types
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

// OpenGL 4.6 function pointers
typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLVERTEXATTRIBDIVISORPROC)(GLuint index, GLuint divisor);
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
typedef void (APIENTRY *PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
typedef void (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
typedef void (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint program);

// Global function pointers
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
static PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
static PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC glBufferData = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
static PFNGLVERTEXATTRIBDIVISORPROC glVertexAttribDivisor = nullptr;
static PFNGLCREATESHADERPROC glCreateShader = nullptr;
static PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
static PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
static PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
static PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
static PFNGLATTACHSHADERPROC glAttachShader = nullptr;
static PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
static PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
static PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
static PFNGLDELETESHADERPROC glDeleteShader = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
static PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;
static PFNGLUNIFORM1FPROC glUniform1f = nullptr;
static PFNGLDRAWARRAYSINSTANCEDPROC glDrawArraysInstanced = nullptr;
static PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
static PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
static PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;

// External CUDA kernel launch function
extern "C" void launchPopulateInstanceDataKernel(
    const PhysicsManager::Circle* d_circles,
    uint32_t count,
    void* d_instanceData,
    uint32_t* d_visibleIndices,
    uint32_t* d_visibleCount,
    float zoomFactor,
    float frustumLeft,
    float frustumRight,
    float frustumBottom,
    float frustumTop,
    cudaStream_t stream
);

// Load OpenGL functions
static bool loadGLFunctions() {
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glfwGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glfwGetProcAddress("glBindVertexArray");
    glGenBuffers = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)glfwGetProcAddress("glEnableVertexAttribArray");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)glfwGetProcAddress("glVertexAttribPointer");
    glVertexAttribDivisor = (PFNGLVERTEXATTRIBDIVISORPROC)glfwGetProcAddress("glVertexAttribDivisor");
    glCreateShader = (PFNGLCREATESHADERPROC)glfwGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)glfwGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)glfwGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)glfwGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glfwGetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)glfwGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)glfwGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)glfwGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glfwGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glfwGetProcAddress("glGetProgramInfoLog");
    glUseProgram = (PFNGLUSEPROGRAMPROC)glfwGetProcAddress("glUseProgram");
    glDeleteShader = (PFNGLDELETESHADERPROC)glfwGetProcAddress("glDeleteShader");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)glfwGetProcAddress("glGetUniformLocation");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)glfwGetProcAddress("glUniformMatrix4fv");
    glUniform1f = (PFNGLUNIFORM1FPROC)glfwGetProcAddress("glUniform1f");
    glDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC)glfwGetProcAddress("glDrawArraysInstanced");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glfwGetProcAddress("glDeleteVertexArrays");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glfwGetProcAddress("glDeleteBuffers");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glfwGetProcAddress("glDeleteProgram");
    
    return glGenVertexArrays && glBindVertexArray && glGenBuffers && glBindBuffer &&
           glBufferData && glEnableVertexAttribArray && glVertexAttribPointer &&
           glVertexAttribDivisor && glCreateShader && glShaderSource && glCompileShader &&
           glGetShaderiv && glGetShaderInfoLog && glCreateProgram && glAttachShader &&
           glLinkProgram && glGetProgramiv && glGetProgramInfoLog && glUseProgram &&
           glDeleteShader && glGetUniformLocation && glUniformMatrix4fv && glUniform1f &&
           glDrawArraysInstanced && glDeleteVertexArrays && glDeleteBuffers && glDeleteProgram;
}

bool RenderingManager::initialize(uint32_t maxCircles, GLFWwindow* window) {
    std::cout << "Initializing rendering manager..." << std::endl;
    
    window_ = window;
    maxCircles_ = maxCircles;
    
    // Make context current
    glfwMakeContextCurrent(window);
    
    // Load OpenGL functions
    if (!loadGLFunctions()) {
        std::cerr << "Failed to load OpenGL functions" << std::endl;
        return false;
    }
    
    // Get window size for viewport
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    // Configure viewport
    glViewport(0, 0, width, height);
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    // Enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Set clear color
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    
    // Create orthographic projection matrix
    // Assuming arena is 10000x10000, center at origin
    float leftPlane = -5000.0f;
    float rightPlane = 5000.0f;
    float bottomPlane = -5000.0f;
    float topPlane = 5000.0f;
    float nearPlane = -1.0f;
    float farPlane = 1.0f;
    
    // Orthographic projection matrix
    projectionMatrix_[0] = 2.0f / (rightPlane - leftPlane);
    projectionMatrix_[1] = 0.0f;
    projectionMatrix_[2] = 0.0f;
    projectionMatrix_[3] = 0.0f;
    
    projectionMatrix_[4] = 0.0f;
    projectionMatrix_[5] = 2.0f / (topPlane - bottomPlane);
    projectionMatrix_[6] = 0.0f;
    projectionMatrix_[7] = 0.0f;
    
    projectionMatrix_[8] = 0.0f;
    projectionMatrix_[9] = 0.0f;
    projectionMatrix_[10] = -2.0f / (farPlane - nearPlane);
    projectionMatrix_[11] = 0.0f;
    
    projectionMatrix_[12] = -(rightPlane + leftPlane) / (rightPlane - leftPlane);
    projectionMatrix_[13] = -(topPlane + bottomPlane) / (topPlane - bottomPlane);
    projectionMatrix_[14] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    projectionMatrix_[15] = 1.0f;
    
    std::cout << "OpenGL context configured successfully" << std::endl;
    std::cout << "Viewport: " << width << "x" << height << std::endl;
    
    // Create CUDA stream for rendering
    cudaError_t err = cudaStreamCreate(&renderStream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Setup shaders and geometry
    setupShaders();
    setupGeometry();
    
    // Setup CUDA-OpenGL interop
    if (!setupCUDAInterop()) {
        std::cerr << "Failed to setup CUDA-OpenGL interop" << std::endl;
        return false;
    }
    
    std::cout << "Rendering manager initialized successfully" << std::endl;
    return true;
}

void RenderingManager::setupShaders() {
    std::cout << "Setting up shaders..." << std::endl;
    
    // Load vertex shader
    std::ifstream vertFile("shaders/circle.vert");
    if (!vertFile.is_open()) {
        std::cerr << "Failed to open vertex shader file" << std::endl;
        return;
    }
    std::stringstream vertStream;
    vertStream << vertFile.rdbuf();
    std::string vertSource = vertStream.str();
    const char* vertSourceCStr = vertSource.c_str();
    
    // Load fragment shader
    std::ifstream fragFile("shaders/circle.frag");
    if (!fragFile.is_open()) {
        std::cerr << "Failed to open fragment shader file" << std::endl;
        return;
    }
    std::stringstream fragStream;
    fragStream << fragFile.rdbuf();
    std::string fragSource = fragStream.str();
    const char* fragSourceCStr = fragSource.c_str();
    
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertSourceCStr, nullptr);
    glCompileShader(vertexShader);
    
    // Check vertex shader compilation
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
        return;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragSourceCStr, nullptr);
    glCompileShader(fragmentShader);
    
    // Check fragment shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
        return;
    }
    
    // Link shader program
    shaderProgram_ = glCreateProgram();
    glAttachShader(shaderProgram_, vertexShader);
    glAttachShader(shaderProgram_, fragmentShader);
    glLinkProgram(shaderProgram_);
    
    // Check linking
    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram_, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
        return;
    }
    
    // Delete shaders (no longer needed after linking)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    std::cout << "Shaders compiled and linked successfully" << std::endl;
}

void RenderingManager::setupGeometry() {
    std::cout << "Setting up geometry..." << std::endl;
    
    // Create VAO
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    
    // Quad vertices (2 triangles forming a square from -1 to 1)
    float quadVertices[] = {
        // Position (x, y)
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f
    };
    
    // Create quad VBO
    glGenBuffers(1, &quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    // Set vertex attribute (location 0: position)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    
    // Create instance VBO (will be populated by CUDA)
    // Instance data structure: position (2), radius (1), lod (1), uvs (4), health (1), padding (2)
    // Total: 11 floats per instance
    size_t instanceDataSize = maxCircles_ * 11 * sizeof(float);
    glGenBuffers(1, &instanceVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_);
    glBufferData(GL_ARRAY_BUFFER, instanceDataSize, nullptr, GL_DYNAMIC_DRAW);
    
    // Set instance attributes
    // Location 1: instancePosition (vec2)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);
    
    // Location 2: instanceRadius (float)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(2 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    
    // Location 3: instanceUVs (vec4)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(4 * sizeof(float)));
    glVertexAttribDivisor(3, 1);
    
    // Location 4: instanceLOD (float)
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glVertexAttribDivisor(4, 1);
    
    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    std::cout << "Geometry setup complete" << std::endl;
}

bool RenderingManager::setupCUDAInterop() {
    std::cout << "Setting up CUDA-OpenGL interop..." << std::endl;
    
    // Register the instance VBO with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cudaVBOResource_,
        instanceVBO_,
        cudaGraphicsMapFlagsWriteDiscard
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to register VBO with CUDA: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate frustum culling buffers
    err = cudaMalloc(&d_visibleIndices_, maxCircles_ * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate visible indices buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_visibleCount_, sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate visible count buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate host memory for visible count
    err = cudaMallocHost(&h_visibleCount_, sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host visible count: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "CUDA-OpenGL interop configured successfully" << std::endl;
    return true;
}

void RenderingManager::updateInstanceData(const PhysicsManager::Circle* circles, uint32_t count, float zoomFactor) {
    // Map OpenGL VBO to CUDA
    cudaError_t err = cudaGraphicsMapResources(1, &cudaVBOResource_, renderStream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to map graphics resource: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Get mapped pointer
    void* d_instanceData;
    size_t numBytes;
    err = cudaGraphicsResourceGetMappedPointer(&d_instanceData, &numBytes, cudaVBOResource_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaVBOResource_, renderStream_);
        return;
    }
    
    // Define frustum (for now, use full arena)
    float frustumLeft = -5000.0f;
    float frustumRight = 5000.0f;
    float frustumBottom = -5000.0f;
    float frustumTop = 5000.0f;
    
    // Launch CUDA kernel to populate instance data with frustum culling
    launchPopulateInstanceDataKernel(
        circles,
        count,
        d_instanceData,
        d_visibleIndices_,
        d_visibleCount_,
        zoomFactor,
        frustumLeft,
        frustumRight,
        frustumBottom,
        frustumTop,
        renderStream_
    );
    
    // Copy visible count back to host
    err = cudaMemcpyAsync(h_visibleCount_, d_visibleCount_, sizeof(uint32_t), 
                          cudaMemcpyDeviceToHost, renderStream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy visible count: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Unmap VBO
    err = cudaGraphicsUnmapResources(1, &cudaVBOResource_, renderStream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to unmap graphics resource: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Synchronize stream
    cudaStreamSynchronize(renderStream_);
}

void RenderingManager::render(const PhysicsManager::Circle* circles, uint32_t count, 
                              float zoomFactor, const TextureAtlasManager& atlasManager) {
    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Update instance data with frustum culling
    updateInstanceData(circles, count, zoomFactor);
    
    // Use shader program
    glUseProgram(shaderProgram_);
    
    // Set uniforms
    GLint projLoc = glGetUniformLocation(shaderProgram_, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projectionMatrix_);
    
    GLint zoomLoc = glGetUniformLocation(shaderProgram_, "zoomFactor");
    glUniform1f(zoomLoc, zoomFactor);
    
    // TODO: Bind texture atlas (will be implemented when texture atlas manager is complete)
    // GLint texLoc = glGetUniformLocation(shaderProgram_, "atlasTexture");
    // glUniform1i(texLoc, 0);
    // glActiveTexture(GL_TEXTURE_2D);
    // glBindTexture(GL_TEXTURE_2D, atlasManager.getOpenGLTextureID());
    
    // Bind VAO and draw
    glBindVertexArray(vao_);
    
    // Draw instanced (6 vertices per quad, h_visibleCount_ instances)
    uint32_t visibleCount = *h_visibleCount_;
    if (visibleCount > 0) {
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, visibleCount);
    }
    
    glBindVertexArray(0);
    
    // Swap buffers
    glfwSwapBuffers(window_);
}

uint32_t RenderingManager::selectLODLevel(float screenRadius) {
    // LOD levels based on screen-space size
    if (screenRadius > 64.0f) return 0;  // Full resolution
    if (screenRadius > 32.0f) return 1;  // 1/2 resolution
    if (screenRadius > 16.0f) return 2;  // 1/4 resolution
    return 3;  // 1/8 resolution
}

void RenderingManager::cleanup() {
    std::cout << "Cleaning up rendering manager..." << std::endl;
    
    // Unregister CUDA graphics resource
    if (cudaVBOResource_) {
        cudaGraphicsUnregisterResource(cudaVBOResource_);
    }
    
    // Free CUDA buffers
    if (d_visibleIndices_) {
        cudaFree(d_visibleIndices_);
    }
    if (d_visibleCount_) {
        cudaFree(d_visibleCount_);
    }
    if (h_visibleCount_) {
        cudaFreeHost(h_visibleCount_);
    }
    
    // Destroy CUDA stream
    if (renderStream_) {
        cudaStreamDestroy(renderStream_);
    }
    
    // Delete OpenGL resources
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
    }
    if (quadVBO_) {
        glDeleteBuffers(1, &quadVBO_);
    }
    if (instanceVBO_) {
        glDeleteBuffers(1, &instanceVBO_);
    }
    if (shaderProgram_) {
        glDeleteProgram(shaderProgram_);
    }
    
    std::cout << "Rendering manager cleaned up" << std::endl;
}
