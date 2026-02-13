#include <cuda_runtime.h>
#include "gpu_rhs_mem.h"
//------init constant memory---------

// 1-----for compute_rhs_bssn---------
__constant__ Meta metac;
__constant__ int ex_c[3];
__constant__ double T_c;
__constant__ int Symmetry_c;
__constant__ int Lev_c;
__constant__ int co_c;
__constant__ double eps_c;
// local
__constant__ double dX; // dX,dY,dZ
__constant__ double dY;
__constant__ double dZ;
__constant__ double ZEO = 1.0;
__constant__ double ONE = 1.0;
__constant__ double TWO = 2.0;
__constant__ double FOUR = 4.0;
__constant__ double EIGHT = 8.0;
__constant__ double HALF = 0.5;
__constant__ double THR = 3.0;
__constant__ double SYM = 1.0;
__constant__ double ANTI = -1.0;
__constant__ double FF = 0.75;
__constant__ double eta = 2.0;
__constant__ double F1o3;
__constant__ double F2o3;
__constant__ double F3o2 = 1.5;
__constant__ double F1o6;
__constant__ double F8 = 8.0;
__constant__ double F16 = 16.0;
__constant__ double PI;
/*__constant__ double SSS[3] = {1,1,1};
__constant__ double AAS[3] = {-1,-1,1};
__constant__ double ASA[3] = {-1,1,-1};
__constant__ double SAA[3] = {1,-1,-1};
__constant__ double ASS[3] = {-1,1,1};
__constant__ double SAS[3] = {1,-1,1};
__constant__ double SSA[3] = {1,1,-1};*/

// 2--------for fderivs------------
__constant__ int ijk_min[3];
__constant__ int ijk_min2[3];
__constant__ int ijk_min3[3];
__constant__ int ijk_max[3];
__constant__ double d12dxyz[3];
__constant__ double d2dxyz[3];

// 3--------for fdderivs------------
__constant__ double Sdxdx;
__constant__ double Sdydy;
__constant__ double Sdzdz;
__constant__ double Fdxdx;
__constant__ double Fdydy;
__constant__ double Fdzdz;
__constant__ double Sdxdy;
__constant__ double Sdxdz;
__constant__ double Sdydz;
__constant__ double Fdxdy;
__constant__ double Fdxdz;
__constant__ double Fdydz;

// my own
__constant__ int STEP_SIZE;
/*__constant__ int MATRIX_SIZE;
__constant__ int MATRIX_SIZE_FH;
__constant__ int SQUARE_SIZE;
__constant__ int SQUARE_SIZE_FH;
__constant__ int LINE_SIZE_FH;*/

__constant__ int _1D_SIZE[4]; // start from 0 !!
__constant__ int _2D_SIZE[4]; ////start from 0 !!
__constant__ int _3D_SIZE[4]; ////start from 0 !!

#if (GAUGE == 6 || GAUGE == 7)
__constant__ int BHN;
__constant__ double Porg[9];
__constant__ double Mass[3];
__constant__ double /*r1,r2*/, M, A, /*w1,w2 (== 12)*/, C1, C2;
#endif