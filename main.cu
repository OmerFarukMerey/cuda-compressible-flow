#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// =============================================================================
// Physical Constants
// =============================================================================
#define Lx      10          // Domain length (m)
#define rho0    1.3         // Reference density (kg/m^3)
#define Mc      0.8         // Mach number
#define vt      1.2         // Velocity transition factor
#define T       (273.15+25) // Temperature (K)
#define gamma   1.4         // Heat capacity ratio
#define Rg      8.314       // Universal gas constant (J/(mol*K))
#define mmol    29E-3       // Molar mass of air (kg/mol)
#define mu_visc 1.85E-5     // Dynamic viscosity (Pa*s)

// Temperature visualization: weight for blending derived temperature deviation
// into the output scalar. T_loc = p*mmol/(rho*Rg); deviation from T0 is shown.
#define T_VIS_GAIN 2.0f     // tunable (e.g. 1.0-4.0)

// Output mode: 1 = colored temperature heatmap (JET, red=hot, blue=cold)
//              0 = grayscale blend (velocity + pressure grad + temperature dev)
#define TEMP_COLORMAP 1
#define T_LO 250.0f         // colormap low  end (K) -> blue  (cold wake ~ -23 C)
#define T_HI 350.0f         // colormap high end (K) -> red   (hot bow shock ~ +77 C)

// Isentropic constant: A = (rho0/mmol * Rg * T) / (rho0/mmol)^gamma
#define Ap ((rho0 / mmol * Rg * T) / (powf(rho0 / mmol, gamma)))

// =============================================================================
// CUDA Block Configuration
// =============================================================================
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

// =============================================================================
// Kernel: sim - Main fluid dynamics simulation step
// Updates velocity (vx, vy), density (rho), and pressure (p)
// Uses shared memory tiling with 1-cell halo for stencil operations
// =============================================================================
__global__ void sim(int Nx, int Ny,
                    float *vxa, float *vya, float *rhoa, float *pa,
                    float *objeta, float dt, float a)
{
    float dl     = Lx / (float)Nx;
    float inv2dl = 1.0f / (2.0f * dl);
    float invdl2 = 1.0f / (dl * dl);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Shared memory tiles with halo
    __shared__ float tile_vx [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_vy [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_p  [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_rho[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_obj[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int idx = i * Ny + j;

    // Load central cell
    if (i < Nx && j < Ny) {
        tile_vx [ty][tx] = vxa[idx];
        tile_vy [ty][tx] = vya[idx];
        tile_p  [ty][tx] = pa[idx];
        tile_rho[ty][tx] = rhoa[idx];
        tile_obj[ty][tx] = objeta[idx];
    }

    // Load halo cells (block borders)
    if (threadIdx.x == 0 && j > 0)
        tile_vx[ty][0] = vxa[i * Ny + (j - 1)];
    if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
        tile_vx[ty][BLOCK_SIZE_X + 1] = vxa[i * Ny + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        tile_vx[0][tx] = vxa[(i - 1) * Ny + j];
    if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
        tile_vx[BLOCK_SIZE_Y + 1][tx] = vxa[(i + 1) * Ny + j];

    if (threadIdx.x == 0 && j > 0)
        tile_vy[ty][0] = vya[i * Ny + (j - 1)];
    if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
        tile_vy[ty][BLOCK_SIZE_X + 1] = vya[i * Ny + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        tile_vy[0][tx] = vya[(i - 1) * Ny + j];
    if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
        tile_vy[BLOCK_SIZE_Y + 1][tx] = vya[(i + 1) * Ny + j];

    if (threadIdx.x == 0 && j > 0)
        tile_rho[ty][0] = rhoa[i * Ny + (j - 1)];
    if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
        tile_rho[ty][BLOCK_SIZE_X + 1] = rhoa[i * Ny + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        tile_rho[0][tx] = rhoa[(i - 1) * Ny + j];
    if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
        tile_rho[BLOCK_SIZE_Y + 1][tx] = rhoa[(i + 1) * Ny + j];

    if (threadIdx.x == 0 && j > 0)
        tile_p[ty][0] = pa[i * Ny + (j - 1)];
    if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
        tile_p[ty][BLOCK_SIZE_X + 1] = pa[i * Ny + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        tile_p[0][tx] = pa[(i - 1) * Ny + j];
    if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
        tile_p[BLOCK_SIZE_Y + 1][tx] = pa[(i + 1) * Ny + j];

    __syncthreads();

    // Interior cells: compute physics update
    if (i > 0 && j > 0 && i < Nx - 1 && j < Ny - 1) {
        // Read neighbors from shared memory
        float vx  = tile_vx[ty][tx];
        float vxh = tile_vx[ty][tx + 1];
        float vxb = tile_vx[ty][tx - 1];
        float vxg = tile_vx[ty + 1][tx];
        float vxd = tile_vx[ty - 1][tx];

        float vy  = tile_vy[ty][tx];
        float vyh = tile_vy[ty][tx + 1];
        float vyb = tile_vy[ty][tx - 1];
        float vyg = tile_vy[ty + 1][tx];
        float vyd = tile_vy[ty - 1][tx];

        float rho  = tile_rho[ty][tx];
        float rhoh = tile_rho[ty][tx + 1];
        float rhob = tile_rho[ty][tx - 1];
        float rhog = tile_rho[ty + 1][tx];
        float rhod = tile_rho[ty - 1][tx];

        float ph = tile_p[ty][tx + 1];
        float pb = tile_p[ty][tx - 1];
        float pg = tile_p[ty + 1][tx];
        float pd = tile_p[ty - 1][tx];

        float objet = tile_obj[ty][tx];

        // Spatial derivatives (2nd-order central differences)
        float dvx_dx = (vxg - vxd) * inv2dl;
        float dvx_dy = (vxh - vxb) * inv2dl;
        float dvy_dx = (vyg - vyd) * inv2dl;
        float dvy_dy = (vyh - vyb) * inv2dl;

        float invrho = 1.0f / rho;

        // Update pressure (isentropic equation of state)
        pa[idx] = Ap * __expf(__logf(rho / mmol) * gamma);

        // Update vx: advection + viscous diffusion + pressure gradient
        float lap_vx = (-4.0f * vx + vxg + vxd + vxh + vxb) * invdl2;
        float dp_dx  = (pg - pd) * inv2dl;
        vxa[idx] += dt * (-vx * dvx_dx
                          - vy * dvx_dy
                          + mu_visc * invrho * lap_vx
                          - invrho * dp_dx);

        // Update vy: advection + viscous diffusion + pressure gradient
        float lap_vy = (-4.0f * vy + vyg + vyd + vyh + vyb) * invdl2;
        float dp_dy  = (ph - pb) * inv2dl;
        vya[idx] += dt * (-vy * dvy_dy
                          - vx * dvy_dx
                          + mu_visc * invrho * lap_vy
                          - invrho * dp_dy);

        // Update density (continuity equation)
        rho += -dt * inv2dl * (rhog * vxg - rhod * vxd
                             + rhoh * vyh - rhob * vyb);
        rhoa[idx] = min(max(rho, 0.1f * rho0), 20.0f * rho0);

        // Solid body: reset density and zero velocity
        if (objet == 0) {
            rhoa[idx] = rho0;
        }
        vxa[idx] *= objet;
        vya[idx] *= objet;
    }

    // Boundary conditions: accelerate flow at domain edges
    if (i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1) {
        vya[idx] += a * dt;
    }
}

// =============================================================================
// Kernel: median - 3x3 median filter for numerical noise suppression
// =============================================================================
__global__ void median(int Nx, int Ny, float *A, float *out)
{
    __shared__ float tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load central cell
    if (i < Nx && j < Ny) {
        tile[ty][tx] = A[i * Ny + j];
    }

    // Load halo (block borders)
    if (i > 0 && j > 0 && i < Nx - 1 && j < Ny - 1) {
        if (threadIdx.x == 0 && j > 0)
            tile[ty][0] = A[i * Ny + (j - 1)];
        if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
            tile[ty][BLOCK_SIZE_X + 1] = A[i * Ny + (j + 1)];
        if (threadIdx.y == 0 && i > 0)
            tile[0][tx] = A[(i - 1) * Ny + j];
        if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
            tile[BLOCK_SIZE_Y + 1][tx] = A[(i + 1) * Ny + j];

        // Corner halos (for complete 3x3 stencil)
        if (threadIdx.x == 0 && threadIdx.y == 0)
            tile[0][0] = A[(i - 1) * Ny + (j - 1)];
        if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE_Y - 1)
            tile[BLOCK_SIZE_Y + 1][0] = A[(i + 1) * Ny + (j - 1)];
        if (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == 0)
            tile[0][BLOCK_SIZE_X + 1] = A[(i - 1) * Ny + (j + 1)];
        if (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == BLOCK_SIZE_Y - 1)
            tile[BLOCK_SIZE_Y + 1][BLOCK_SIZE_X + 1] = A[(i + 1) * Ny + (j + 1)];
    }

    __syncthreads();

    if (i > 0 && j > 0 && i < Nx - 1 && j < Ny - 1) {
        // Gather 3x3 neighborhood
        float window[9] = {
            tile[ty - 1][tx - 1], tile[ty - 1][tx], tile[ty - 1][tx + 1],
            tile[ty    ][tx - 1], tile[ty    ][tx], tile[ty    ][tx + 1],
            tile[ty + 1][tx - 1], tile[ty + 1][tx], tile[ty + 1][tx + 1]
        };

        // Insertion sort to find median
        for (int k = 1; k < 9; ++k) {
            float key = window[k];
            int m = k - 1;
            while (m >= 0 && window[m] > key) {
                window[m + 1] = window[m];
                m--;
            }
            window[m + 1] = key;
        }

        out[i * Ny + j] = window[4];
    }
}

// =============================================================================
// Kernel: aff - Compute visualization scalar
// Combines velocity magnitude and pressure gradient for rendering
// =============================================================================
__global__ void aff(int Nx, int Ny,
                    float *vxa, float *vya, float *pa, float *rhoa,
                    float *objeta, float *resultat, float dt, float a)
{
    float P0      = 101300.0f;
    float v0      = vxa[0];
    float invv0p2 = 1.0f / (v0 * v0);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_vx [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_vy [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_p  [BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ float tile_obj[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int tx  = threadIdx.x + 1;
    int ty  = threadIdx.y + 1;
    int idx = i * Ny + j;

    // Load central cell
    if (i < Nx && j < Ny) {
        tile_vx [ty][tx] = vxa[idx];
        tile_vy [ty][tx] = vya[idx];
        tile_p  [ty][tx] = pa[idx];
        tile_obj[ty][tx] = objeta[idx];
    }

    // Load pressure halo
    if (threadIdx.x == 0 && j > 0)
        tile_p[ty][0] = pa[i * Ny + (j - 1)];
    if (threadIdx.x == BLOCK_SIZE_X - 1 && j < Ny - 1)
        tile_p[ty][BLOCK_SIZE_X + 1] = pa[i * Ny + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        tile_p[0][tx] = pa[(i - 1) * Ny + j];
    if (threadIdx.y == BLOCK_SIZE_Y - 1 && i < Nx - 1)
        tile_p[BLOCK_SIZE_Y + 1][tx] = pa[(i + 1) * Ny + j];

    __syncthreads();

    if (i > 0 && j > 0 && i < Nx - 1 && j < Ny - 1) {
        float vx  = tile_vx[ty][tx];
        float vy  = tile_vy[ty][tx];
        float obj = tile_obj[ty][tx];

        float ph = tile_p[ty][tx + 1];
        float pb = tile_p[ty][tx - 1];
        float pg = tile_p[ty + 1][tx];
        float pd = tile_p[ty - 1][tx];

        float dpdx = pg - pd;
        float dpdy = ph - pb;

        // Derived temperature (ideal gas): T_loc = p*mmol/(rho*Rg).
        float Tloc = tile_p[ty][tx] * mmol / (rhoa[idx] * Rg);  // tile_p[ty][tx] == pa[idx]

#if TEMP_COLORMAP
        // Normalized temperature in [0,1] for a JET colormap (red = hot).
        // Solid body cells evaluate to T0 (ambient) so the silhouette stays visible.
        resultat[idx] = (Tloc - T_LO) / (T_HI - T_LO);
        (void)vx; (void)vy; (void)obj; (void)dpdx; (void)dpdy; (void)v0; (void)invv0p2;
#else
        // Grayscale: velocity magnitude + pressure gradient + temperature deviation
        resultat[idx] = (1.0f - obj)
                      + 0.5f * sqrtf(((vx - v0) * (vx - v0) + vy * vy) * invv0p2);
        resultat[idx] += (1.0f - obj)
                       + sqrtf(dpdx * dpdx + dpdy * dpdy) / P0;
        resultat[idx] += obj * T_VIS_GAIN * fabsf(Tloc - T) / T;
#endif
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[])
{
    // Parse command-line argument for image path
    const char *image_path = "concordecote.png";
    if (argc > 1) {
        image_path = argv[1];
    }

    // Compute derived physical quantities
    float P0 = (float)(rho0 / mmol * Rg * T);
    float c  = sqrtf(gamma * P0 / rho0);          // Speed of sound
    float v  = Mc * c;                              // Initial flow velocity

    // Simulation parameters
    int   R    = 2160;                              // Grid resolution (rows)
    int   vsim = (argc > 3) ? atoi(argv[3]) : 2000; // Display/save interval
    float dt   = Lx / (10.0f * R * (v + vt * c));  // CFL-based timestep
    int   Nt   = (argc > 2) ? atoi(argv[2]) : (int)(1e3 * vsim); // Total iterations

    // Acceleration to reach target velocity over simulation time
    float a = vt * c / (float)(Nt * dt);

    cudaSetDevice(0);

    // Load obstacle geometry from image
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not load '" << image_path << "'" << endl;
        return -1;
    }
    resize(image, image, Size((int)(R * 16 / 9.0), R), 0, 0, INTER_NEAREST);
    GaussianBlur(image, image, Size(5, 5), 0);

    // Force binary: remove gray/anti-aliased pixels
    threshold(image, image, 128, 255, THRESH_BINARY);

    // Auto-detect: if more black than white, image is inverted — flip it
    if (countNonZero(image) < image.rows * image.cols / 2) {
        bitwise_not(image, image);
    }

    int Ny = image.cols;
    int Nx = image.rows;

    // Allocate host memory
    float *vx       = new float[Nx * Ny];
    float *vy       = new float[Nx * Ny];
    float *p        = new float[Nx * Ny];
    float *objet    = new float[Nx * Ny];
    float *resultat = new float[Nx * Ny];
    float *rho      = new float[Nx * Ny];

    // Convert obstacle image to float mask (1 = fluid, 0 = solid)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            objet[i * Ny + j] = (float)image.data[i * Ny + j] / 255.0f;
        }
    }

    // CUDA grid/block configuration
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid(Ny / BLOCK_SIZE_X, Nx / BLOCK_SIZE_Y);

    // Initialize fields
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            vx[i * Ny + j]       = v * objet[i * Ny + j];
            vy[i * Ny + j]       = 0.0f;
            p[i * Ny + j]        = P0;
            resultat[i * Ny + j] = 0.0f;
            rho[i * Ny + j]      = (float)rho0;
        }
    }

    // Allocate device memory
    float *vxa, *vya, *pa, *objeta, *rhoa, *medstock, *temp, *result;
    cudaMalloc(&vxa,     Nx * Ny * sizeof(float));
    cudaMalloc(&vya,     Nx * Ny * sizeof(float));
    cudaMalloc(&rhoa,    Nx * Ny * sizeof(float));
    cudaMalloc(&pa,      Nx * Ny * sizeof(float));
    cudaMalloc(&objeta,  Nx * Ny * sizeof(float));
    cudaMalloc(&result,  Nx * Ny * sizeof(float));
    cudaMalloc(&medstock, 9 * Nx * Ny * sizeof(float));
    cudaMalloc(&temp,    Nx * Ny * sizeof(float));

    cout << "CUDA init: " << cudaGetErrorString(cudaGetLastError()) << endl;

    // Copy data to device
    cudaMemcpy(vxa,    vx,    Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vya,    vy,    Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pa,     p,     Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(objeta, objet, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rhoa,   rho,   Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);

    // Main simulation loop
    int t0 = time(NULL);
    Mat plot, plot0;

    for (int it = 0; it < Nt; it++) {
        sim<<<dimGrid, dimBlock>>>(Nx, Ny, vya, vxa, rhoa, pa, objeta, dt, a);
        v += a * dt;

        // Apply median filter every 10 steps for stability
        if (it % 10 == 0) {
            median<<<dimGrid, dimBlock>>>(Nx, Ny, vxa, vxa);
            median<<<dimGrid, dimBlock>>>(Nx, Ny, vya, vya);
            median<<<dimGrid, dimBlock>>>(Nx, Ny, pa, pa);
            median<<<dimGrid, dimBlock>>>(Nx, Ny, rhoa, rhoa);
        }

        // Display and save results
        if (it % vsim == 0) {
            float pourcent = 100.0f * (float)it / (float)Nt;
            int t1 = time(NULL);
            int Ttravail = (int)((t1 - t0) * (float)Nt / ((float)(it + 1)));
            int trestant = (int)(Ttravail * (100.0f - pourcent) / 100.0f);
            int h  = trestant / 3600;
            int mi = (trestant - h * 3600) / 60;
            int s  = trestant - h * 3600 - mi * 60;

            cout << "\n--- Progress ---" << endl;
            cout << pourcent << "% | Iteration " << it << " / " << Nt << endl;
            cout << "Time remaining: " << h << "h " << mi << "m " << s << "s" << endl;
            cout << "Velocity: " << v << " m/s" << endl;
            cout << "Reynolds number: " << (rho0 * v * Lx) / (40.0f * mu_visc) << endl;
            cout << "CFL stability: " << (1.0f - v * dt / (Lx / (float)Nx)) << endl;
            cout << "Physical time: " << Nt * dt << " s" << endl;
            cout << "CUDA: " << cudaGetErrorString(cudaGetLastError()) << endl;

            // Compute visualization field
            aff<<<dimGrid, dimBlock>>>(Nx, Ny, vxa, vya, pa, rhoa, objeta, result, dt, a);
            cudaMemcpy(resultat, result, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
            plot0 = Mat(Nx, Ny, CV_32F, resultat);

            // Save frame to disk
            if (it % (1 * vsim) == 0) {
                plot0.convertTo(plot, CV_8U, 255);   // result in [0,1] -> [0,255], clamped
#if TEMP_COLORMAP
                Mat plotc;
                applyColorMap(plot, plotc, COLORMAP_JET);   // blue = cold, red = hot
                imwrite("resultatsim/" + to_string(it / vsim) + ".jpg", plotc);
                resize(plotc, plotc, Size((int)(900 * 16 / 9.0), 900), 0, 0, INTER_AREA);
                imshow("Temperature (JET: blue=cold, red=hot)", plotc);
#else
                imwrite("resultatsim/" + to_string(it / vsim) + ".jpg", plot);
                resize(plot0, plot0, Size((int)(900 * 16 / 9.0), 900), 0, 0, INTER_AREA);
                imshow("Compressible", plot0);
#endif
                waitKey(1);
            }
        }
    }

    // Cleanup
    cudaFree(vxa);
    cudaFree(vya);
    cudaFree(pa);
    cudaFree(objeta);
    cudaFree(rhoa);
    cudaFree(result);
    cudaFree(medstock);
    cudaFree(temp);

    delete[] vx;
    delete[] vy;
    delete[] p;
    delete[] objet;
    delete[] resultat;
    delete[] rho;

    return 0;
}
