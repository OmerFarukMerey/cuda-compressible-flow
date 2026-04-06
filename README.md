# ✈️ GPU-Accelerated 2D Compressible Flow Simulator

**CUDA-accelerated 2D compressible Navier-Stokes solver — simulating supersonic flow and shock wave formation around aerodynamic bodies, inspired by Concorde.**

![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

`shockwave` is a GPU-accelerated 2D Computational Fluid Dynamics (CFD) simulator built with CUDA. It numerically solves the **compressible Navier-Stokes equations** on a 2D structured grid using finite difference methods, and visualizes the transition from subsonic to supersonic flow (Mach < 1 → Mach > 1) around user-defined aerodynamic bodies.

The project is motivated by the aerodynamics of the **Concorde** — one of the most iconic supersonic aircraft ever built — and aims to numerically reproduce the flow phenomena observable in its real wind tunnel tests.

> 🎬 Inspiration: [Concorde Wind Tunnel Test Archive Footage](https://www.youtube.com/watch?v=DD53Er62GrE)

---

## Features

- 🔴 **Compressible Navier-Stokes solver** — continuity, momentum, and energy equations on a 2D grid
- ⚡ **CUDA GPU acceleration** — shared memory tiling with halo regions for stencil computations
- 🌡️ **Temperature field evolution** — energy equation extension for aerodynamic heating visualization
- 🛩️ **Image-based geometry** — define any aerodynamic body using a grayscale PNG silhouette mask
- 📊 **CPU vs GPU benchmarking** — systematic performance comparison across grid resolutions and block configurations
- 🎨 **Real-time visualization** — live OpenCV rendering of flow fields (velocity, pressure, density, temperature)

---

## Physics

The solver numerically integrates the 2D compressible Navier-Stokes equations:

**Continuity:**
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$

**Momentum:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \frac{\mu}{\rho}\nabla^2 \mathbf{u}$$

**Equation of state (isentropic):**
$$p = A \cdot \rho^{\gamma}$$

**Energy (extension):**
$$\frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla)T = \alpha \nabla^2 T - \frac{(\gamma - 1)T}{\rho} \nabla \cdot (\rho \mathbf{u})$$

Spatial derivatives are computed using **2nd-order central finite differences**. A **median filter** is applied periodically to suppress numerical instabilities.

---

## CUDA Architecture

| Kernel | Description |
|--------|-------------|
| `sim` | Main physics update — velocity, density, pressure fields |
| `energy` | Temperature field evolution (energy equation) |
| `median` | 3×3 median filter for numerical smoothing |
| `aff` | Visualization — computes display field from flow variables |

All kernels use **shared memory tiling with halo regions** (block size: 32×16) to minimize global memory traffic for stencil-based computations.

```
Grid Cell (i, j)
    ┌─────────────┐
    │  tile[ty][tx] ← shared memory
    │  halo loaded from neighbors
    │  __syncthreads()
    │  stencil computed entirely from SMEM
    └─────────────┘
```

---

## Getting Started

### Prerequisites

- NVIDIA GPU (Compute Capability ≥ 8.6 recommended)
- CUDA Toolkit 12.x
- OpenCV 4.x
- C++17 compiler

### Build

```bash
nvcc main.cu -o shockwave \
  $(pkg-config --cflags opencv4) \
  $(pkg-config --libs opencv4) \
  -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
  -arch=compute_86
```

### Run

```bash
./shockwave
```

Place your aerodynamic body silhouette as `body.png` (grayscale, white = fluid, black = solid) in the working directory. The Concorde profile used in development is included in `assets/`.

---

## Configuration

Key parameters in `config.h`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Lx` | Domain length (m) | `10` |
| `Mc` | Initial Mach number | `0.8` |
| `vt` | Target Mach multiplier | `1.2` |
| `BLOCK_SIZE_X` | CUDA block width | `32` |
| `BLOCK_SIZE_Y` | CUDA block height | `16` |
| `Nt` | Number of timesteps | `2×10⁶` |

---

## Benchmarking

Performance is evaluated by comparing GPU vs CPU execution time across:

- Grid resolutions: 540×960, 1080×1920, 2160×3840
- Block configurations: 8×8, 16×16, 32×16, 32×32
- Fields: velocity only vs. full energy equation extension

Results and analysis are reported in the [project report](report/).

---

## Results

Simulation outputs are saved to `results/` as JPEG frames and can be compiled into a video:

```bash
ffmpeg -framerate 30 -i results/%d.jpg -c:v libx264 output.mp4
```

---

## Project Structure

```
shockwave/
├── main.cu              # Entry point, simulation loop
├── kernels/
│   ├── sim.cuh          # Physics kernel
│   ├── energy.cuh       # Temperature field kernel
│   ├── median.cuh       # Smoothing kernel
│   └── aff.cuh          # Visualization kernel
├── assets/
│   └── concorde.png     # Concorde silhouette mask
├── results/             # Simulation output frames
├── report/              # Project report (IEEE format)
├── config.h             # Simulation parameters
└── README.md
```

---

## References

1. Anderson, J. D. (2003). *Modern Compressible Flow*. McGraw-Hill.
2. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
3. Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors*. Morgan Kaufmann.
4. [Concorde Wind Tunnel Test Footage](https://www.youtube.com/watch?v=DD53Er62GrE)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*METU — Applied Parallel Programming on GPU — Spring 2024-2025*
