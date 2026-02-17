#include "enforce_algebra.h"

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_rhs_mem.h"
// #include "bssn_gpu_rhs.h"
// #include "bssn_gpu_rhs_constant.h"

// __global__ void enforce_ga_kernel(
//     int n,
//     double* dxx, double* gxy, double* gxz,
//     double* dyy, double* gyz, double* dzz,
//     double* Axx, double* Axy, double* Axz,
//     double* Ayy, double* Ayz, double* Azz
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     while (tid < n) {
//         double gxx = dxx[tid] + ONE;
//         double gyy = dyy[tid] + ONE;
//         double gzz = dzz[tid] + ONE;
//         double gxyv = gxy[tid];
//         double gxzv = gxz[tid];
//         double gyzv = gyz[tid];

//         double detg = gxx * gyy * gzz + gxyv * gyzv * gxzv + gxzv * gxyv * gyzv
//                     - gxzv * gyy * gxzv - gxyv * gxyv * gzz - gxx * gyzv * gyzv;

//         double scale = ONE / pow(detg, F1o3);

//         gxx *= scale; gxyv *= scale; gxzv *= scale;
//         gyy *= scale; gyzv *= scale; gzz *= scale;

//         dxx[tid] = gxx - ONE;
//         dyy[tid] = gyy - ONE;
//         dzz[tid] = gzz - ONE;
//         gxy[tid] = gxyv;
//         gxz[tid] = gxzv;
//         gyz[tid] = gyzv;

//         double gupxx =   ( gyy * gzz - gyzv * gyzv );
//         double gupxy = - ( gxyv * gzz - gyzv * gxzv );
//         double gupxz =   ( gxyv * gyzv - gyy * gxzv );
//         double gupyy =   ( gxx * gzz - gxzv * gxzv );
//         double gupyz = - ( gxx * gyzv - gxyv * gxzv );
//         double gupzz =   ( gxx * gyy - gxyv * gxyv );

//         double trA = gupxx * Axx[tid] + gupyy * Ayy[tid] + gupzz * Azz[tid]
//                    + TWO * (gupxy * Axy[tid] + gupxz * Axz[tid] + gupyz * Ayz[tid]);

//         Axx[tid] -= F1o3 * gxx * trA;
//         Axy[tid] -= F1o3 * gxyv * trA;
//         Axz[tid] -= F1o3 * gxzv * trA;
//         Ayy[tid] -= F1o3 * gyy * trA;
//         Ayz[tid] -= F1o3 * gyzv * trA;
//         Azz[tid] -= F1o3 * gzz * trA;

//         tid += STEP_SIZE;
//     }
// }

// void gpu_enforce_ga(GPU_RHS_CONTEXT &ctx) {
//     cudaSetDevice(DEVICE_ID);
//     Meta *meta = gpu_get_meta();
//     int n = ctx.ex[0] * ctx.ex[1] * ctx.ex[2];
//     enforce_ga_kernel<<<RHS_GRID_DIM, RHS_BLOCK_DIM>>>(
//         n,
//         Mh_ dxx, Mh_ gxy, Mh_ gxz, Mh_ dyy, Mh_ gyz, Mh_ dzz,
//         Mh_ Axx, Mh_ Axy, Mh_ Axz, Mh_ Ayy, Mh_ Ayz, Mh_ Azz
//     );
// }