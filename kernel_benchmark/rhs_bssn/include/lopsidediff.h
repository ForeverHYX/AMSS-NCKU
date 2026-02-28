#ifndef LOPSIDEDIFF_H
#define LOPSIDEDIFF_H

#include <cuda_runtime.h>

__device__ double d_lopsided_point(
    const int ex[3], const double* f,
    const double* f_rhs, const double* Sfx, const double* Sfy, const double* Sfz,
    const double* X, const double* Y, const double* Z,
    int symmetry, double SYM1, double SYM2, double SYM3,
    int i, int j, int k
);

#endif /* LOPSIDEDIFF_H */