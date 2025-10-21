# Assets Directory

## Players

Place player avatar images in the `players/` subdirectory.

### Supported Formats
- JPG
- PNG

### Recommended Specifications
- Resolution: 256x256 pixels (will be automatically resized if different)
- File naming: Use descriptive names (e.g., `player_001.jpg`, `avatar_alice.png`)
- Maximum count: Up to 1,000,000 unique avatars supported

### Notes
- Images will be automatically packed into texture atlases
- Mipmaps will be generated for LOD rendering
- Missing or corrupt images will use a default placeholder texture
- Larger images will consume more memory during loading but will be compressed for GPU storage
