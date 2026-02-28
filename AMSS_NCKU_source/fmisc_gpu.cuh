#ifndef FMISC_GPU_CUH
#define FMISC_GPU_CUH

__device__ __forceinline__ double d_symmetry_bd_0b(
    int ord, 
    int extc0, int extc1, int extc2,
    const double* __restrict__ func,
    int i, int j, int k, // 0-based coordinates
    double SoA0, double SoA1, double SoA2
) {
    if (i < -ord || i >= extc0) return 0.0;
    if (j < -ord || j >= extc1) return 0.0;
    if (k < -ord || k >= extc2) return 0.0;

    double factor = 1.0;

    if (i < 0) { i = -i - 1; factor *= SoA0; }
    if (j < 0) { j = -j - 1; factor *= SoA1; }
    if (k < 0) { k = -k - 1; factor *= SoA2; }

    if (i >= extc0 || j >= extc1 || k >= extc2) return 0.0;

    return func[(k * extc1 + j) * extc0 + i] * factor;
}

#endif // FMISC_GPU_CUH