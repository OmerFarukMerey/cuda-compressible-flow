#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
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

// Isentropic constant
#define Ap ((rho0 / mmol * Rg * T) / (powf(rho0 / mmol, gamma)))

// =============================================================================
// CPU: sim - Main fluid dynamics simulation step
// =============================================================================
void sim_cpu(int Nx, int Ny,
             float *vxa, float *vya, float *rhoa, float *pa,
             float *objeta, float dt, float a)
{
    float dl     = Lx / (float)Nx;
    float inv2dl = 1.0f / (2.0f * dl);
    float invdl2 = 1.0f / (dl * dl);

    // Interior cells
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            int idx = i * Ny + j;

            float vx  = vxa[idx];
            float vxh = vxa[i * Ny + (j + 1)];
            float vxb = vxa[i * Ny + (j - 1)];
            float vxg = vxa[(i + 1) * Ny + j];
            float vxd = vxa[(i - 1) * Ny + j];

            float vy  = vya[idx];
            float vyh = vya[i * Ny + (j + 1)];
            float vyb = vya[i * Ny + (j - 1)];
            float vyg = vya[(i + 1) * Ny + j];
            float vyd = vya[(i - 1) * Ny + j];

            float rho  = rhoa[idx];
            float rhoh = rhoa[i * Ny + (j + 1)];
            float rhob = rhoa[i * Ny + (j - 1)];
            float rhog = rhoa[(i + 1) * Ny + j];
            float rhod = rhoa[(i - 1) * Ny + j];

            float ph = pa[i * Ny + (j + 1)];
            float pb = pa[i * Ny + (j - 1)];
            float pg = pa[(i + 1) * Ny + j];
            float pd = pa[(i - 1) * Ny + j];

            float objet = objeta[idx];

            // Spatial derivatives (2nd-order central differences)
            float dvx_dx = (vxg - vxd) * inv2dl;
            float dvx_dy = (vxh - vxb) * inv2dl;
            float dvy_dx = (vyg - vyd) * inv2dl;
            float dvy_dy = (vyh - vyb) * inv2dl;

            float invrho = 1.0f / rho;

            // Update pressure (isentropic equation of state)
            pa[idx] = Ap * expf(logf(rho / mmol) * gamma);

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
    }

    // Boundary conditions: accelerate flow at domain edges
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1) {
                vya[i * Ny + j] += a * dt;
            }
        }
    }
}

// =============================================================================
// CPU: median - 3x3 median filter for numerical noise suppression
// =============================================================================
void median_cpu(int Nx, int Ny, float *A, float *out)
{
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            // Gather 3x3 neighborhood
            float window[9] = {
                A[(i - 1) * Ny + (j - 1)], A[(i - 1) * Ny + j], A[(i - 1) * Ny + (j + 1)],
                A[ i      * Ny + (j - 1)], A[ i      * Ny + j], A[ i      * Ny + (j + 1)],
                A[(i + 1) * Ny + (j - 1)], A[(i + 1) * Ny + j], A[(i + 1) * Ny + (j + 1)]
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
}

// =============================================================================
// CPU: aff - Compute visualization scalar
// =============================================================================
void aff_cpu(int Nx, int Ny,
             float *vxa, float *vya, float *pa,
             float *objeta, float *resultat)
{
    float P0      = 101300.0f;
    float v0      = vxa[0];
    float invv0p2 = 1.0f / (v0 * v0);

    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            int idx = i * Ny + j;

            float vx  = vxa[idx];
            float vy  = vya[idx];
            float obj = objeta[idx];

            float ph = pa[i * Ny + (j + 1)];
            float pb = pa[i * Ny + (j - 1)];
            float pg = pa[(i + 1) * Ny + j];
            float pd = pa[(i - 1) * Ny + j];

            float dpdx = pg - pd;
            float dpdy = ph - pb;

            resultat[idx] = (1.0f - obj)
                          + 0.5f * sqrtf(((vx - v0) * (vx - v0) + vy * vy) * invv0p2);
            resultat[idx] += (1.0f - obj)
                           + sqrtf(dpdx * dpdx + dpdy * dpdy) / P0;
        }
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
    int   vsim = 2000;                              // Display interval
    float dt   = Lx / (10.0f * R * (v + vt * c));  // CFL-based timestep
    int   Nt   = (int)(1e3 * vsim);                 // Total iterations

    // Acceleration to reach target velocity over simulation time
    float a = vt * c / (float)(Nt * dt);

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

    cout << "CPU simulation: " << Nx << " x " << Ny << " grid, " << Nt << " iterations" << endl;

    // Allocate memory
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

    // Main simulation loop
    int t0 = time(NULL);
    Mat plot, plot0;

    for (int it = 0; it < Nt; it++) {
        sim_cpu(Nx, Ny, vy, vx, rho, p, objet, dt, a);
        v += a * dt;

        // Apply median filter every 10 steps for stability
        if (it % 10 == 0) {
            median_cpu(Nx, Ny, vx, vx);
            median_cpu(Nx, Ny, vy, vy);
            median_cpu(Nx, Ny, p, p);
            median_cpu(Nx, Ny, rho, rho);
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

            // Compute visualization field
            aff_cpu(Nx, Ny, vx, vy, p, objet, resultat);
            plot0 = Mat(Nx, Ny, CV_32F, resultat);

            // Save frame to disk
            if (it % (1 * vsim) == 0) {
                plot0.convertTo(plot, CV_8U, 255);
                imwrite("resultatsim_cpu/" + to_string(it / vsim) + ".jpg", plot);
                resize(plot0, plot0, Size((int)(900 * 16 / 9.0), 900), 0, 0, INTER_AREA);
                imshow("Compressible (CPU)", plot0);
                waitKey(1);
            }
        }
    }

    // Cleanup
    delete[] vx;
    delete[] vy;
    delete[] p;
    delete[] objet;
    delete[] resultat;
    delete[] rho;

    return 0;
}
