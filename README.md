# DXF Assembly Container Packer

A cross-platform desktop application that loads 3D DXF assembly files (at arbitrary angles),
automatically normalises their orientation, packs them into a shipping container, inserts
dunnage spacer logs, and visualises the result in an interactive 3D viewport.

---

## Features

- **DXF loading** — 3DFACE entities (including nested in INSERT blocks), MESH, POLYFACE MESH
- **OBB normalisation** — every assembly is auto-aligned before packing, regardless of input orientation
- **Extreme-Point Best-Fit Decreasing** 3D bin packing with 24 axis-aligned rotations
- **Gravity constraint** — no floating items; everything rests on the floor or another item
- **Spacer log (dunnage) insertion** — two logs under each floor item + horizontal separators
- **Fine-rotation mode** — optional 15° Z-step pass for tighter packing (slower)
- **Interactive 3D viewport** — pyqtgraph OpenGL with mouse rotate / zoom / pan and item selection
- **Export** — DXF, JSON manifest, CSV flat table, PDF report with screenshot
- **Shipping container presets** — 20ft, 40ft, 40ft High Cube, or custom dimensions

---

## Installation

### Requirements
- Python 3.11 or newer
- Windows / macOS / Linux

### Steps

```bash
# 1. Clone / extract the project
cd DXFtoContainer

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on `python-fcl`** — trimesh's `CollisionManager` uses `python-fcl` for mesh-level
> collision detection. Installing `python-fcl` on Windows can be tricky because it requires a
> compiled C++ library. If installation fails:
> - Try `pip install python-fcl` in an environment with Visual C++ build tools
> - Or install via conda: `conda install -c conda-forge python-fcl`
> - **The app works without it** — it falls back to AABB-overlap collision checks automatically.

---

## Running

```bash
python main.py
```

---

## Generating Sample DXFs

```bash
python samples/generate_samples.py
```

This produces three sample files in `samples/`:

| File | Description |
|------|-------------|
| `box_simple.dxf` | 800×600×500 mm axis-aligned box |
| `l_shape_tilted.dxf` | L-shape tilted 13° Z + 7° X (tests OBB normalisation) |
| `thin_beam_45deg.dxf` | 1800×120×120 mm beam at 45° Z (tests fine-rotation mode) |

---

## Building a Standalone Executable

**Windows:**
```bat
build.bat
```

**macOS / Linux:**
```bash
bash build.sh
```

Output: `dist/DXFPacker.exe` (Windows) or `dist/DXFPacker` (macOS/Linux)

---

## Quick Start Workflow

1. Launch the app: `python main.py`
2. Click **Add DXF…** (or File → Add DXF…) to load one or more assembly files
3. Set quantity, weight, and tipping permission per assembly in the left panel
4. Choose a container preset (right panel) or enter custom dimensions
5. Adjust clearance, log cross-section, and optimization mode
6. Click **PACK**
7. Rotate/zoom the 3D viewport to inspect the result
8. Export via the right panel buttons

---

## Screenshot

![DXF Packer viewport](screenshot_placeholder.png)

*(Replace with an actual screenshot after first run)*

---

## Known Limitations

- **python-fcl** may require manual installation on Windows (see note above).
  Without it, collision checking uses AABB overlap (less precise but always available).
- Very large assemblies (>100k triangles each) will slow packing; consider simplifying meshes.
- Fine-rotation mode (15° steps) is significantly slower for large item counts.
- The packer uses bounding-box clearance, not actual mesh clearance, for speed.
- PDF screenshot requires an active OpenGL context; headless/CI environments may produce
  a blank screenshot image.
- DXF export writes the inverse-normalised mesh geometry, so the output DXF correctly
  reflects the original model coordinates at the packed position.

---

## Architecture

```
dxf_packer/
├── dxf_loader.py   — DXF → trimesh.Trimesh
├── assembly.py     — Assembly dataclass + OBB normalisation
├── container.py    — Container dataclass + presets
├── rotations.py    — 24 axis-aligned rotations + fine Z rotations
├── packer.py       — Extreme-Point BFFD algorithm + gravity + dunnage
├── exporter.py     — DXF / JSON / CSV / PDF export
├── utils.py        — Shared helpers (colours, wireframe geometry)
└── ui/
    ├── main_window.py  — QMainWindow, threading, export wiring
    ├── viewport.py     — pyqtgraph OpenGL 3D viewport
    ├── left_panel.py   — Assembly list with per-row controls
    └── right_panel.py  — Container/packing settings + Pack button
```
