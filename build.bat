@echo off
REM Build DXF Packer into a standalone Windows executable using PyInstaller
REM Run from the project root: build.bat

echo Building DXF Assembly Container Packer...

pyinstaller ^
    --name "DXFPacker" ^
    --onefile ^
    --windowed ^
    --icon NONE ^
    --add-data "samples;samples" ^
    --hidden-import "pyqtgraph.opengl" ^
    --hidden-import "OpenGL" ^
    --hidden-import "OpenGL.GL" ^
    --hidden-import "OpenGL.arrays" ^
    --hidden-import "OpenGL.arrays.vbo" ^
    --hidden-import "ezdxf" ^
    --hidden-import "trimesh" ^
    --hidden-import "trimesh.collision" ^
    --hidden-import "reportlab" ^
    --hidden-import "scipy" ^
    --hidden-import "shapely" ^
    main.py

if %ERRORLEVEL% == 0 (
    echo.
    echo Build successful! Executable: dist\DXFPacker.exe
) else (
    echo.
    echo Build FAILED. Check output above for errors.
)
