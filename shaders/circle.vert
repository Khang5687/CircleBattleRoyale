#version 460 core

// Vertex attributes
layout(location = 0) in vec2 vertexPosition;  // Circle quad vertices (-1 to 1)

// Instance attributes
layout(location = 1) in vec2 instancePosition;
layout(location = 2) in float instanceRadius;
layout(location = 3) in vec4 instanceUVs;  // u0, v0, u1, v1
layout(location = 4) in float instanceLOD;

// Outputs
out vec2 fragTexCoord;
out float fragLOD;

// Uniforms
uniform mat4 projection;
uniform float zoomFactor;

void main() {
    // Scale vertex by radius and zoom
    vec2 scaledPos = vertexPosition * instanceRadius * zoomFactor;
    vec2 worldPos = instancePosition + scaledPos;
    
    gl_Position = projection * vec4(worldPos, 0.0, 1.0);
    
    // Interpolate UV coordinates
    fragTexCoord = mix(instanceUVs.xy, instanceUVs.zw, (vertexPosition + 1.0) * 0.5);
    fragLOD = instanceLOD;
}
