#ifndef GPU_CONSTANT_H
#define GPU_CONSTANT_H

#include "gpu_mem.h"
#include <cuda_runtime.h>
//------init constant memory---------

// 1-----for compute_rhs_bssn---------
extern __constant__ Meta metac;
extern __constant__ int ex_c[3];
extern __constant__ double T_c;
extern __constant__ int Symmetry_c;
extern __constant__ int Lev_c;
extern __constant__ int co_c;
extern __constant__ double eps_c;
// local
extern __constant__ double dX; // dX,dY,dZ
extern __constant__ double dY;
extern __constant__ double dZ;
extern __constant__ double ZEO;
extern __constant__ double ONE;
extern __constant__ double TWO;
extern __constant__ double FOUR;
extern __constant__ double EIGHT;
extern __constant__ double HALF;
extern __constant__ double THR;
extern __constant__ double SYM;
extern __constant__ double ANTI ;
extern __constant__ double FF ;
extern __constant__ double eta;
extern __constant__ double F1o3;
extern __constant__ double F2o3;
extern __constant__ double F3o2;
extern __constant__ double F1o6;
extern __constant__ double F8;
extern __constant__ double F16 ;
extern __constant__ double PI;
/*__constant__ double SSS[3] = {1,1,1};
__constant__ double AAS[3] = {-1,-1,1};
__constant__ double ASA[3] = {-1,1,-1};
__constant__ double SAA[3] = {1,-1,-1};
__constant__ double ASS[3] = {-1,1,1};
__constant__ double SAS[3] = {1,-1,1};
__constant__ double SSA[3] = {1,1,-1};*/

// 2--------for fderivs------------
extern __constant__ int ijk_min[3];
extern __constant__ int ijk_min2[3];
extern __constant__ int ijk_min3[3];
extern __constant__ int ijk_max[3];
extern __constant__ double d12dxyz[3];
extern __constant__ double d2dxyz[3];

// 3--------for fdderivs------------
extern __constant__ double Sdxdx;
extern __constant__ double Sdydy;
extern __constant__ double Sdzdz;
extern __constant__ double Fdxdx;
extern __constant__ double Fdydy;
extern __constant__ double Fdzdz;
extern __constant__ double Sdxdy;
extern __constant__ double Sdxdz;
extern __constant__ double Sdydz;
extern __constant__ double Fdxdy;
extern __constant__ double Fdxdz;
extern __constant__ double Fdydz;

// my own
extern __constant__ int STEP_SIZE;
/*__constant__ int MATRIX_SIZE;
__constant__ int MATRIX_SIZE_FH;
__constant__ int SQUARE_SIZE;
__constant__ int SQUARE_SIZE_FH;
__constant__ int LINE_SIZE_FH;*/

extern __constant__ int _1D_SIZE[4]; // start from 0 !!
extern __constant__ int _2D_SIZE[4]; ////start from 0 !!
extern __constant__ int _3D_SIZE[4]; ////start from 0 !!

#if (GAUGE == 6 || GAUGE == 7)
__constant__ int BHN;
__constant__ double Porg[9];
__constant__ double Mass[3];
__constant__ double /*r1,r2*/, M, A, /*w1,w2 (== 12)*/, C1, C2;
#endif

#endif