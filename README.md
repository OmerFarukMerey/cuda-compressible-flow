# GPU-Accelerated 2D Compressible Flow Simulator

**CUDA-powered 2D compressible Navier-Stokes solver for simulating supersonic flow and shock wave formation around aerodynamic bodies.**

![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

A GPU-accelerated 2D CFD simulator that solves the **compressible Navier-Stokes equations** on a structured grid using finite differences. It visualizes the transition from subsonic to supersonic flow (Mach 0.8 to Mach 1.2+) around any aircraft silhouette — Concorde, Tupolev, Boeing, or your own custom shape.

Just drop in a PNG silhouette and watch shock waves form in real time.

---

## Demo

Shock wave formation around a Concorde silhouette as flow transitions from Mach 0.8 to Mach 1.2+:

<video src="sample_simulation_videos/concorde.mp4" controls width="100%"></video>

> GitHub does not auto-play raw `.mp4` files committed to the repo. [Click here to download or view the video directly](sample_simulation_videos/concorde.mp4).

---

## Features

- **Compressible Navier-Stokes solver** on a 2160x3840 grid (~8.3M cells)
- **CUDA GPU acceleration** with shared memory tiling and halo regions
- **Image-based geometry** — any grayscale PNG silhouette works as input
- **Auto image preprocessing** — handles inverted colors and gray backgrounds automatically
- **Real-time OpenCV visualization** of flow fields
- **CPU reference implementation** for performance comparison
- **Video generation** from simulation frames via Python script

---

## Physics

The solver integrates the 2D compressible Navier-Stokes equations:

**Continuity:**
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

**Momentum:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \frac{\mu}{\rho}\nabla^2 \mathbf{u}$$

**Equation of state (isentropic):**
$$p = A \cdot \rho^{\gamma}$$

Spatial derivatives use **2nd-order central finite differences** with **explicit Euler** time integration. A **3x3 median filter** is applied every 10 steps to suppress numerical oscillations.

---

## CUDA Architecture

Three GPU kernels, all using **shared memory tiling with 1-cell halo regions** (block size: 32x16 threads):

| Kernel | Purpose |
|--------|---------|
| `sim` | Main physics — updates velocity, density, and pressure fields |
| `median` | 3x3 median filter for numerical stability |
| `aff` | Computes visualization scalar (velocity magnitude + pressure gradient) |

Each thread processes one grid cell. Neighbor data is loaded into shared memory tiles to minimize global memory access during stencil computations.

---

## Getting Started

### Prerequisites

- NVIDIA GPU (Compute Capability 8.6+)
- CUDA Toolkit 12.x+
- OpenCV 4.x
- C++17 compiler (MSVC, GCC, or Clang)

### Build & Run (Windows with vcpkg)

**GPU version:**
```bash
bash build_gpu.sh images/concordecote.png
```

**CPU version:**
```bash
bash build_cpu.sh images/concordecote.png
```

Or compile manually:
```bash
nvcc main.cu -o compressible.exe \
  -I"C:/vcpkg/installed/x64-windows/include/opencv4" \
  -L"C:/vcpkg/installed/x64-windows/lib" \
  -lopencv_core4 -lopencv_highgui4 -lopencv_imgcodecs4 -lopencv_imgproc4 \
  -arch=compute_120 -allow-unsupported-compiler

mkdir -p resultatsim
./compressible.exe images/concordecote.png
```

### Build & Run (Linux)

```bash
nvcc main.cu -o compressible \
  $(pkg-config --cflags --libs opencv4) \
  -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
  -arch=compute_86

mkdir -p resultatsim
./compressible images/concordecote.png
```

> Adjust `-arch=compute_XX` to match your GPU. Use `compute_86` for RTX 30-series, `compute_89` for RTX 40-series, `compute_120` for RTX 50-series.

---

## Input Images

The simulator accepts any grayscale PNG as input. The image is used as an obstacle mask:

- **Black pixels** = solid body (aircraft)
- **White pixels** = fluid (air)

The code automatically handles inverted images (white-on-black) and removes gray artifacts via thresholding.

Several silhouettes are included in `images/`:

| Image | Aircraft |
|-------|----------|
| `concordecote.png` | Concorde side profile |
| `tupolev.png` | Tupolev Tu-144 |
| `boeing777800.png` | Boeing 777-800 |
| `concorde-noseup.png` | Concorde nose-up angle |

---

## Creating a Video

After the simulation saves frames to `resultatsim/`, create a video:

```bash
python make_video.py
```

Or with custom options:
```bash
python make_video.py resultatsim simulation.mp4
```

---

## Configuration

Key parameters are defined as macros in `main.cu`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Lx` | Domain length (m) | `10` |
| `Mc` | Initial Mach number | `0.8` |
| `vt` | Velocity transition factor | `1.2` |
| `gamma` | Heat capacity ratio | `1.4` |
| `mu_visc` | Dynamic viscosity (Pa*s) | `1.85e-5` |
| `BLOCK_SIZE_X` | CUDA block width | `32` |
| `BLOCK_SIZE_Y` | CUDA block height | `16` |

Grid resolution is set to 2160x3840 by default (R=2160, 16:9 aspect ratio).

---

## Project Structure

```
CUDA_project/
├── main.cu              # GPU simulation (CUDA)
├── main_cpu.cpp          # CPU reference implementation
├── build_gpu.sh          # Build & run GPU version
├── build_cpu.sh          # Build & run CPU version
├── make_video.py         # Convert frames to video
├── images/               # Aircraft silhouette masks
│   ├── concordecote.png
│   ├── tupolev.png
│   ├── boeing777800.png
│   └── ...
└── resultatsim/          # Output frames (generated)
```

---

## References

1. Anderson, J. D. (2003). *Modern Compressible Flow*. McGraw-Hill.
2. Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors*. Morgan Kaufmann.
3. Micikevicius, P. (2009). *3D Finite Difference Computation on GPUs using CUDA*. GPGPU-2.

---

## License

MIT License
