// =============================================================================
// temp_demo.cpp  —  Standalone temperature demonstration (CPU)
//
// Runs the same compressible-flow physics as main_cpu.cpp, then renders the
// DERIVED temperature field  T = p * mmol / (rho * Rg)  as a JET heatmap and
// prints the real min/max/mean temperature in Kelvin and Celsius.
//
// This file is only for *showing* that a temperature field exists; it does not
// change the main solver. Build:
//   cl /EHsc /O2 /Fe:temp_demo.exe temp_demo.cpp /I"...opencv4" /link ... opencv_*.lib
//   temp_demo.exe images/concordecote.png 240 3000
// =============================================================================
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// ---- Physical constants (identical to main_cpu.cpp) ----
#define Lx      10
#define rho0    1.3
#define Mc      0.8
#define vt      1.2
#define T       (273.15+25)   // reference temperature T0 (K)
#define gamma   1.4
#define Rg      8.314
#define mmol    29E-3
#define mu_visc 1.85E-5
#define Ap ((rho0 / mmol * Rg * T) / (powf(rho0 / mmol, gamma)))

// ---- sim step (copied verbatim from main_cpu.cpp) ----
void sim_cpu(int Nx, int Ny,
             float *vxa, float *vya, float *rhoa, float *pa,
             float *objeta, float dt, float a)
{
    float dl     = Lx / (float)Nx;
    float inv2dl = 1.0f / (2.0f * dl);
    float invdl2 = 1.0f / (dl * dl);

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

            float dvx_dx = (vxg - vxd) * inv2dl;
            float dvx_dy = (vxh - vxb) * inv2dl;
            float dvy_dx = (vyg - vyd) * inv2dl;
            float dvy_dy = (vyh - vyb) * inv2dl;

            float invrho = 1.0f / rho;

            pa[idx] = Ap * expf(logf(rho / mmol) * gamma);

            float lap_vx = (-4.0f * vx + vxg + vxd + vxh + vxb) * invdl2;
            float dp_dx  = (pg - pd) * inv2dl;
            vxa[idx] += dt * (-vx * dvx_dx - vy * dvx_dy
                              + mu_visc * invrho * lap_vx - invrho * dp_dx);

            float lap_vy = (-4.0f * vy + vyg + vyd + vyh + vyb) * invdl2;
            float dp_dy  = (ph - pb) * inv2dl;
            vya[idx] += dt * (-vy * dvy_dy - vx * dvy_dx
                              + mu_visc * invrho * lap_vy - invrho * dp_dy);

            rho += -dt * inv2dl * (rhog * vxg - rhod * vxd
                                 + rhoh * vyh - rhob * vyb);
            rhoa[idx] = fminf(fmaxf(rho, 0.1f * (float)rho0), 20.0f * (float)rho0);

            if (objet == 0) rhoa[idx] = rho0;
            vxa[idx] *= objet;
            vya[idx] *= objet;
        }
    }

    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            if (i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1)
                vya[i * Ny + j] += a * dt;
}

// ---- median filter (copied verbatim) ----
void median_cpu(int Nx, int Ny, float *A, float *out)
{
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            float window[9] = {
                A[(i - 1) * Ny + (j - 1)], A[(i - 1) * Ny + j], A[(i - 1) * Ny + (j + 1)],
                A[ i      * Ny + (j - 1)], A[ i      * Ny + j], A[ i      * Ny + (j + 1)],
                A[(i + 1) * Ny + (j - 1)], A[(i + 1) * Ny + j], A[(i + 1) * Ny + (j + 1)]
            };
            for (int k = 1; k < 9; ++k) {
                float key = window[k];
                int m = k - 1;
                while (m >= 0 && window[m] > key) { window[m + 1] = window[m]; m--; }
                window[m + 1] = key;
            }
            out[i * Ny + j] = window[4];
        }
    }
}

int main(int argc, char *argv[])
{
    const char *image_path = (argc > 1) ? argv[1] : "images/concordecote.png";
    int R  = (argc > 2) ? atoi(argv[2]) : 240;    // small grid for a fast demo
    int Nt = (argc > 3) ? atoi(argv[3]) : 3000;

    float P0 = (float)(rho0 / mmol * Rg * T);
    float c  = sqrtf(gamma * P0 / rho0);
    float v  = Mc * c;
    float dt = Lx / (10.0f * R * (v + vt * c));
    float a  = vt * c / (float)(Nt * dt);

    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) { cerr << "Could not load " << image_path << endl; return -1; }
    resize(image, image, Size((int)(R * 16 / 9.0), R), 0, 0, INTER_NEAREST);
    GaussianBlur(image, image, Size(5, 5), 0);
    threshold(image, image, 128, 255, THRESH_BINARY);
    if (countNonZero(image) < image.rows * image.cols / 2) bitwise_not(image, image);

    int Ny = image.cols, Nx = image.rows;
    cout << "Temperature demo: " << Nx << " x " << Ny << " grid, "
         << Nt << " iterations, image=" << image_path << endl;
    cout << "Reference (freestream) temperature T0 = " << (float)T
         << " K  (" << (float)T - 273.15f << " C)\n" << endl;

    float *vx  = new float[Nx * Ny];
    float *vy  = new float[Nx * Ny];
    float *p   = new float[Nx * Ny];
    float *obj = new float[Nx * Ny];
    float *rho = new float[Nx * Ny];

    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++) {
            int idx = i * Ny + j;
            obj[idx] = (float)image.data[idx] / 255.0f;
            vx[idx]  = v * obj[idx];
            vy[idx]  = 0.0f;
            p[idx]   = P0;
            rho[idx] = (float)rho0;
        }

    for (int it = 0; it < Nt; it++) {
        sim_cpu(Nx, Ny, vy, vx, rho, p, obj, dt, a);
        v += a * dt;
        if (it % 10 == 0) {
            median_cpu(Nx, Ny, vx, vx);
            median_cpu(Nx, Ny, vy, vy);
            median_cpu(Nx, Ny, p,  p);
            median_cpu(Nx, Ny, rho, rho);
        }
        if (it % 500 == 0) cout << "  iter " << it << " / " << Nt << "  (v=" << v << " m/s)" << endl;
    }

    // ---- Derive the temperature field: T = p * mmol / (rho * Rg) ----
    Mat Tk(Nx, Ny, CV_32F);
    float Tmin =  numeric_limits<float>::max();
    float Tmax = -numeric_limits<float>::max();
    double Tsum = 0.0; long nfluid = 0;
    int imax = 0, jmax = 0, imin = 0, jmin = 0;

    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++) {
            int idx = i * Ny + j;
            float Tloc = p[idx] * (float)mmol / (rho[idx] * (float)Rg);
            Tk.at<float>(i, j) = Tloc;
            if (obj[idx] > 0.5f) {               // fluid cells only for stats
                Tsum += Tloc; nfluid++;
                if (Tloc > Tmax) { Tmax = Tloc; imax = i; jmax = j; }
                if (Tloc < Tmin) { Tmin = Tloc; imin = i; jmin = j; }
            }
        }
    float Tmean = (float)(Tsum / (double)nfluid);

    cout << "\n================ DERIVED TEMPERATURE FIELD ================\n";
    cout << "  T = p * mmol / (rho * Rg)\n";
    cout << "  min  T = " << Tmin  << " K  (" << Tmin  - 273.15f << " C)  at cell (" << imin << "," << jmin << ")\n";
    cout << "  mean T = " << Tmean << " K  (" << Tmean - 273.15f << " C)\n";
    cout << "  max  T = " << Tmax  << " K  (" << Tmax  - 273.15f << " C)  at cell (" << imax << "," << jmax << ")\n";
    cout << "  spread = " << (Tmax - Tmin) << " K\n";
    cout << "===========================================================\n" << endl;

    // ---- Render JET heatmap normalized over [Tmin, Tmax] ----
    Mat T8, Tcolor;
    Tk.convertTo(T8, CV_8U, 255.0 / (Tmax - Tmin), -255.0 * Tmin / (Tmax - Tmin));
    applyColorMap(T8, Tcolor, COLORMAP_JET);   // blue=cold, red=hot
    imwrite("temperature_demo.jpg", Tcolor);
    cout << "Saved heatmap -> temperature_demo.jpg  (blue = cold, red = hot)" << endl;

    delete[] vx; delete[] vy; delete[] p; delete[] obj; delete[] rho;
    return 0;
}
