#!/usr/bin/env bash
# Build DXF Packer into a standalone executable using PyInstaller
# Run from the project root: bash build.sh

set -e
echo "Building DXF Assembly Container Packer..."

pyinstaller \
    --name "DXFPacker" \
    --onefile \
    --windowed \
    --add-data "samples:samples" \
    --hidden-import "pyqtgraph.opengl" \
    --hidden-import "OpenGL" \
    --hidden-import "OpenGL.GL" \
    --hidden-import "OpenGL.arrays" \
    --hidden-import "OpenGL.arrays.vbo" \
    --hidden-import "ezdxf" \
    --hidden-import "trimesh" \
    --hidden-import "trimesh.collision" \
    --hidden-import "reportlab" \
    --hidden-import "scipy" \
    --hidden-import "shapely" \
    main.py

echo ""
echo "Build complete. Executable: dist/DXFPacker"
