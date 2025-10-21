#version 460 core

// Inputs
in vec2 fragTexCoord;
in float fragLOD;

// Output
out vec4 fragColor;

// Uniforms
uniform sampler2D atlasTexture;

void main() {
    // Sample texture with automatic LOD selection
    vec4 texColor = texture(atlasTexture, fragTexCoord);
    
    // Discard pixels outside circle (alpha test)
    if (texColor.a < 0.1) {
        discard;
    }
    
    fragColor = texColor;
}
