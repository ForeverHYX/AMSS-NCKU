#include <cuda_runtime.h>
#include <math.h>

#define MAX_ORDN 6
#ifndef GPU_DEBUG_PRINT
#define GPU_DEBUG_PRINT 0
#endif

__device__ __forceinline__ void gpu_stop() {
#if GPU_STRICT_STOP
    asm("trap;");
#endif
}

__device__ __forceinline__ double X_at_1b(const double* X, int i1b) {
	return X[i1b - 1];
}
__device__ __forceinline__ double f_at_1b(const double* f, const int ex[3], int i1b, int j1b, int k1b) {
	return f[((k1b - 1) * ex[1] + (j1b - 1)) * ex[0] + (i1b - 1)];
}

__device__ void polint(const double* xa, const double* ya, double x, double& y, double& dy, int ordn) {
	double c[MAX_ORDN], d[MAX_ORDN], den[MAX_ORDN], ho[MAX_ORDN];
	int ns = 1;
	double dif = fabs(x - xa[0]);
	for (int m = 0; m < ordn; ++m) {
		c[m] = ya[m];
		d[m] = ya[m];
		ho[m] = xa[m] - x;
		double dift = fabs(x - xa[m]);
		if (dift < dif) { ns = m + 1; dif = dift; }
	}
	y = ya[ns - 1];
	ns = ns - 1;
	for (int m = 1; m < ordn; ++m) {
		for (int i = 0; i < ordn - m; ++i) {
			den[i] = ho[i] - ho[i + m];
			if (den[i] == 0.0) {
#if GPU_DEBUG_PRINT
                printf("failure in polint for point %f\n", x);
                printf("with input points: ");
                for (int t = 0; t < ordn; ++t) printf("%f ", xa[t]);
                printf("\n");
#endif
				y = NAN; dy = NAN; gpu_stop(); return;
			}
			den[i] = (c[i + 1] - d[i]) / den[i];
			d[i] = ho[i + m] * den[i];
			c[i] = ho[i] * den[i];
		}
		if (2 * ns < (ordn - m)) {
			dy = c[ns];
		} else {
			dy = d[ns - 1];
			ns = ns - 1;
		}
		y = y + dy;
	}
}

__device__ void polin3(
	const double* x1a, const double* x2a, const double* x3a,
	const double* ya, double x1, double x2, double x3,
	double& y, double& dy, int ordn
) {
	double yatmp[MAX_ORDN * MAX_ORDN];
	double ymtmp[MAX_ORDN];
	double yntmp[MAX_ORDN];
	double yqtmp[MAX_ORDN];

	for (int i = 0; i < ordn; ++i) {
		for (int j = 0; j < ordn; ++j) {
			for (int k = 0; k < ordn; ++k) {
				yqtmp[k] = ya[(k * ordn + j) * ordn + i];
			}
			polint(x3a, yqtmp, x3, yatmp[j * ordn + i], dy, ordn);
		}
		for (int j = 0; j < ordn; ++j) yntmp[j] = yatmp[j * ordn + i];
		polint(x2a, yntmp, x2, ymtmp[i], dy, ordn);
	}
	polint(x1a, ymtmp, x1, y, dy, ordn);
}

__device__ bool decide3d(
	const int ex[3], const double* f, const double* fpi,
	const int cxB[3], const int cxT[3], const double SoA[3],
	double* ya, int ordn, int Symmetry
) {
	(void)fpi;
	(void)Symmetry;
	bool gont = false;
	int fmin1[3], fmin2[3], fmax1[3], fmax2[3];

	for (int m = 0; m < 3; ++m) {
		if (!(abs(cxB[m]) >= 0)) gont = true;
		if (!(abs(cxT[m]) >= 0)) gont = true;
		fmin1[m] = max(1, cxB[m]);
		fmax1[m] = cxT[m];
		fmin2[m] = cxB[m];
		fmax2[m] = min(0, cxT[m]);
		if ((fmin1[m] <= fmax1[m]) && (fmin1[m] < 1 || fmax1[m] > ex[m])) gont = true;
		if ((fmin2[m] <= fmax2[m]) && (1 - fmax2[m] < 1 || 1 - fmin2[m] > ex[m])) gont = true;
	}
	if (gont) {
#if GPU_DEBUG_PRINT
        printf("error in decide3d\n");
        printf("cxB: %d %d %d, cxT: %d %d %d, ex: %d %d %d\n",
               cxB[0], cxB[1], cxB[2], cxT[0], cxT[1], cxT[2], ex[0], ex[1], ex[2]);
        printf("fmin1: %d %d %d, fmax1: %d %d %d\n",
               fmin1[0], fmin1[1], fmin1[2], fmax1[0], fmax1[1], fmax1[2]);
        printf("fmin2: %d %d %d, fmax2: %d %d %d\n",
               fmin2[0], fmin2[1], fmin2[2], fmax2[0], fmax2[1], fmax2[2]);
#endif
		return true;
	}

	auto idx = [&](int i, int j, int k) {
		return ((k - cxB[2]) * ordn + (j - cxB[1])) * ordn + (i - cxB[0]);
	};

	for (int k = fmin1[2]; k <= fmax1[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, k);
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, k) * SoA[0];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, k) * SoA[1];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, k) * SoA[0] * SoA[1];
		}
	}

	for (int k = fmin2[2]; k <= fmax2[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, 1 - k) * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, 1 - k) * SoA[0] * SoA[2];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, 1 - k) * SoA[1] * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, 1 - k) * SoA[0] * SoA[1] * SoA[2];
		}
	}
	return false;
}

__device__ void global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry
) {
	if (ORDN > MAX_ORDN) { f_int[0] = NAN; gpu_stop(); return; }

	const int NO_SYMM = 0, EQUATORIAL = 1, OCTANT = 2;
	int imin = 1, jmin = 1, kmin = 1;

	double dX = X_at_1b(X, imin + 1) - X_at_1b(X, imin);
	double dY = X_at_1b(Y, jmin + 1) - X_at_1b(Y, jmin);
	double dZ = X_at_1b(Z, kmin + 1) - X_at_1b(Z, kmin);

	double x1a[MAX_ORDN];
	for (int j = 0; j < ORDN; ++j) x1a[j] = (double)j;

	int cxI[3];
	cxI[0] = (int)((x1 - X_at_1b(X, 1)) / dX + 0.4) + 1;
	cxI[1] = (int)((y1 - X_at_1b(Y, 1)) / dY + 0.4) + 1;
	cxI[2] = (int)((z1 - X_at_1b(Z, 1)) / dZ + 0.4) + 1;

	int cxB[3], cxT[3], cmin[3], cmax[3];
	for (int m = 0; m < 3; ++m) {
		cxB[m] = cxI[m] - ORDN / 2 + 1;
		cxT[m] = cxB[m] + ORDN - 1;
		cmin[m] = 1;
		cmax[m] = ex[m];
	}
	if (symmetry == OCTANT && fabs(X_at_1b(X, 1)) < dX) cmin[0] = -ORDN / 2 + 1;
	if (symmetry == OCTANT && fabs(X_at_1b(Y, 1)) < dY) cmin[1] = -ORDN / 2 + 1;
	if (symmetry != NO_SYMM && fabs(X_at_1b(Z, 1)) < dZ) cmin[2] = -ORDN / 2 + 1;

	for (int m = 0; m < 3; ++m) {
		if (cxB[m] < cmin[m]) { cxB[m] = cmin[m]; cxT[m] = cxB[m] + ORDN - 1; }
		if (cxT[m] > cmax[m]) { cxT[m] = cmax[m]; cxB[m] = cxT[m] + 1 - ORDN; }
	}

	double cx[3];
	cx[0] = (cxB[0] > 0) ? (x1 - X_at_1b(X, cxB[0])) / dX : (x1 + X_at_1b(X, 1 - cxB[0])) / dX;
	cx[1] = (cxB[1] > 0) ? (y1 - X_at_1b(Y, cxB[1])) / dY : (y1 + X_at_1b(Y, 1 - cxB[1])) / dY;
	cx[2] = (cxB[2] > 0) ? (z1 - X_at_1b(Z, cxB[2])) / dZ : (z1 + X_at_1b(Z, 1 - cxB[2])) / dZ;

	double ya[MAX_ORDN * MAX_ORDN * MAX_ORDN];
	if (decide3d(ex, f, f, cxB, cxT, SoA, ya, ORDN, symmetry)) {
#if GPU_DEBUG_PRINT
        printf("global_interp position: %f %f %f\n", x1, y1, z1);
        printf("data range: %f %f %f %f %f %f\n",
               X_at_1b(X, 1), X_at_1b(X, ex[0]),
               X_at_1b(Y, 1), X_at_1b(Y, ex[1]),
               X_at_1b(Z, 1), X_at_1b(Z, ex[2]));
#endif
		f_int[0] = NAN;
		gpu_stop();
		return;
	}

	double ddy = 0.0;
	polin3(x1a, x1a, x1a, ya, cx[0], cx[1], cx[2], f_int[0], ddy, ORDN);
}