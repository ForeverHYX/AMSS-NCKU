#include "bssn_gpu_manager.h"

#include "gpu_mem.h"
#include "bssn_gpu.h"
#include "gpu_constant.h"

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

using namespace std;

static Meta * meta = NULL;

Meta * gpu_get_meta() {
	return meta;
}

void gpu_destroy_meta() {	
	if(Mh_ X) cudaFree(Mh_ X);
	if(Mh_ Y) cudaFree(Mh_ Y);
	if(Mh_ Z) cudaFree(Mh_ Z);
	if(Mh_ chi) cudaFree(Mh_ chi);
	if(Mh_ dxx) cudaFree(Mh_ dxx);
	if(Mh_ dyy) cudaFree(Mh_ dyy);
	if(Mh_ dzz) cudaFree(Mh_ dzz);
	if(Mh_ trK) cudaFree(Mh_ trK);
	if(Mh_ gxy) cudaFree(Mh_ gxy);
	if(Mh_ gxz) cudaFree(Mh_ gxz);
	if(Mh_ gyz) cudaFree(Mh_ gyz);
	if(Mh_ Axx) cudaFree(Mh_ Axx);
	if(Mh_ Axy) cudaFree(Mh_ Axy);
	if(Mh_ Axz) cudaFree(Mh_ Axz);
	if(Mh_ Ayz) cudaFree(Mh_ Ayz);
	if(Mh_ Ayy) cudaFree(Mh_ Ayy);
	if(Mh_ Azz) cudaFree(Mh_ Azz);
	if(Mh_ Gamx) cudaFree(Mh_ Gamx);
	if(Mh_ Gamy) cudaFree(Mh_ Gamy);
	if(Mh_ Gamz) cudaFree(Mh_ Gamz);
	if(Mh_ Lap) cudaFree(Mh_ Lap);
	if(Mh_ betax) cudaFree(Mh_ betax);
	if(Mh_ betay) cudaFree(Mh_ betay);
	if(Mh_ betaz) cudaFree(Mh_ betaz);
	if(Mh_ dtSfx) cudaFree(Mh_ dtSfx);
	if(Mh_ dtSfy) cudaFree(Mh_ dtSfy);
	if(Mh_ dtSfz) cudaFree(Mh_ dtSfz);
	if(Mh_ chi_rhs) cudaFree(Mh_ chi_rhs);
	if(Mh_ trK_rhs) cudaFree(Mh_ trK_rhs);
	if(Mh_ gxy_rhs) cudaFree(Mh_ gxy_rhs);
	if(Mh_ gxz_rhs) cudaFree(Mh_ gxz_rhs);
	if(Mh_ gyz_rhs) cudaFree(Mh_ gyz_rhs);
	if(Mh_ Axx_rhs) cudaFree(Mh_ Axx_rhs);
	if(Mh_ Axy_rhs) cudaFree(Mh_ Axy_rhs);
	if(Mh_ Axz_rhs) cudaFree(Mh_ Axz_rhs);
	if(Mh_ Ayz_rhs) cudaFree(Mh_ Ayz_rhs);
	if(Mh_ Ayy_rhs) cudaFree(Mh_ Ayy_rhs);
	if(Mh_ Azz_rhs) cudaFree(Mh_ Azz_rhs);
	if(Mh_ Gamx_rhs) cudaFree(Mh_ Gamx_rhs);
	if(Mh_ Gamy_rhs) cudaFree(Mh_ Gamy_rhs);
	if(Mh_ Gamz_rhs) cudaFree(Mh_ Gamz_rhs);
	if(Mh_ Lap_rhs) cudaFree(Mh_ Lap_rhs);
	if(Mh_ betax_rhs) cudaFree(Mh_ betax_rhs);
	if(Mh_ betay_rhs) cudaFree(Mh_ betay_rhs);  
	if(Mh_ betaz_rhs) cudaFree(Mh_ betaz_rhs);
	if(Mh_ dtSfx_rhs) cudaFree(Mh_ dtSfx_rhs);
	if(Mh_ dtSfy_rhs) cudaFree(Mh_ dtSfy_rhs);
	if(Mh_ dtSfz_rhs) cudaFree(Mh_ dtSfz_rhs);
	if(Mh_ rho) cudaFree(Mh_ rho);
	if(Mh_ Sx) cudaFree(Mh_ Sx);
	if(Mh_ Sy) cudaFree(Mh_ Sy);
	if(Mh_ Sz) cudaFree(Mh_ Sz);
	if(Mh_ Sxx) cudaFree(Mh_ Sxx);
	if(Mh_ Sxy) cudaFree(Mh_ Sxy);
	if(Mh_ Sxz) cudaFree(Mh_ Sxz);
	if(Mh_ Syz) cudaFree(Mh_ Syz);
	if(Mh_ Syy) cudaFree(Mh_ Syy);
	if(Mh_ Szz) cudaFree(Mh_ Szz);
	if(Mh_ Gamxxx) cudaFree(Mh_ Gamxxx);
	if(Mh_ Gamxxy) cudaFree(Mh_ Gamxxy);
	if(Mh_ Gamxxz) cudaFree(Mh_ Gamxxz);
	if(Mh_ Gamxyy) cudaFree(Mh_ Gamxyy);
	if(Mh_ Gamxyz) cudaFree(Mh_ Gamxyz);
	if(Mh_ Gamxzz) cudaFree(Mh_ Gamxzz);
	if(Mh_ Gamyxx) cudaFree(Mh_ Gamyxx);
	if(Mh_ Gamyxy) cudaFree(Mh_ Gamyxy);
	if(Mh_ Gamyxz) cudaFree(Mh_ Gamyxz);
	if(Mh_ Gamyyy) cudaFree(Mh_ Gamyyy);
	if(Mh_ Gamyyz) cudaFree(Mh_ Gamyyz);
	if(Mh_ Gamyzz) cudaFree(Mh_ Gamyzz);
	if(Mh_ Gamzxx) cudaFree(Mh_ Gamzxx);
	if(Mh_ Gamzxy) cudaFree(Mh_ Gamzxy);
	if(Mh_ Gamzxz) cudaFree(Mh_ Gamzxz);
	if(Mh_ Gamzyz) cudaFree(Mh_ Gamzyz);
	if(Mh_ Gamzyy) cudaFree(Mh_ Gamzyy);
	if(Mh_ Gamzzz) cudaFree(Mh_ Gamzzz);
	if(Mh_ Rxx) cudaFree(Mh_ Rxx);
	if(Mh_ Rxy) cudaFree(Mh_ Rxy);
	if(Mh_ Rxz) cudaFree(Mh_ Rxz);
	if(Mh_ Ryy) cudaFree(Mh_ Ryy);
	if(Mh_ Ryz) cudaFree(Mh_ Ryz);
	if(Mh_ Rzz) cudaFree(Mh_ Rzz);
	if(Mh_ ham_Res) cudaFree(Mh_ ham_Res);
	if(Mh_ movx_Res) cudaFree(Mh_ movx_Res);
	if(Mh_ movy_Res) cudaFree(Mh_ movy_Res);
	if(Mh_ movz_Res) cudaFree(Mh_ movz_Res);
	if(Mh_ Gmx_Res) cudaFree(Mh_ Gmx_Res);
	if(Mh_ Gmy_Res) cudaFree(Mh_ Gmy_Res);
	if(Mh_ Gmz_Res) cudaFree(Mh_ Gmz_Res);
	if(Mh_ gxx) cudaFree(Mh_ gxx);
	if(Mh_ gyy) cudaFree(Mh_ gyy);
	if(Mh_ gzz) cudaFree(Mh_ gzz);
	if(Mh_ chix) cudaFree(Mh_ chix);
	if(Mh_ chiy) cudaFree(Mh_ chiy);
	if(Mh_ chiz) cudaFree(Mh_ chiz);
	if(Mh_ gxxx) cudaFree(Mh_ gxxx);
	if(Mh_ gxyx) cudaFree(Mh_ gxyx);
	if(Mh_ gxzx) cudaFree(Mh_ gxzx);
	if(Mh_ gyyx) cudaFree(Mh_ gyyx);
	if(Mh_ gyzx) cudaFree(Mh_ gyzx);
	if(Mh_ gzzx) cudaFree(Mh_ gzzx);
	if(Mh_ gxxy) cudaFree(Mh_ gxxy);
	if(Mh_ gxyy) cudaFree(Mh_ gxyy);
	if(Mh_ gxzy) cudaFree(Mh_ gxzy);
	if(Mh_ gyyy) cudaFree(Mh_ gyyy);
	if(Mh_ gyzy) cudaFree(Mh_ gyzy);
	if(Mh_ gzzy) cudaFree(Mh_ gzzy);
	if(Mh_ gxxz) cudaFree(Mh_ gxxz);
	if(Mh_ gxyz) cudaFree(Mh_ gxyz);
	if(Mh_ gxzz) cudaFree(Mh_ gxzz);
	if(Mh_ gyyz) cudaFree(Mh_ gyyz);
	if(Mh_ gyzz) cudaFree(Mh_ gyzz);
	if(Mh_ gzzz) cudaFree(Mh_ gzzz);
	if(Mh_ Lapx) cudaFree(Mh_ Lapx);
	if(Mh_ Lapy) cudaFree(Mh_ Lapy);
	if(Mh_ Lapz) cudaFree(Mh_ Lapz);
	if(Mh_ betaxx) cudaFree(Mh_ betaxx);
	if(Mh_ betaxy) cudaFree(Mh_ betaxy);
	if(Mh_ betaxz) cudaFree(Mh_ betaxz);
	if(Mh_ betayy) cudaFree(Mh_ betayy);
	if(Mh_ betayz) cudaFree(Mh_ betayz);
	if(Mh_ betazz) cudaFree(Mh_ betazz);
	if(Mh_ betayx) cudaFree(Mh_ betayx);
	if(Mh_ betazy) cudaFree(Mh_ betazy);
	if(Mh_ betazx) cudaFree(Mh_ betazx);
	if(Mh_ Kx) cudaFree(Mh_ Kx);
	if(Mh_ Ky) cudaFree(Mh_ Ky);
	if(Mh_ Kz) cudaFree(Mh_ Kz);
	if(Mh_ Gamxx) cudaFree(Mh_ Gamxx);
	if(Mh_ Gamxy) cudaFree(Mh_ Gamxy);
	if(Mh_ Gamxz) cudaFree(Mh_ Gamxz);
	if(Mh_ Gamyy) cudaFree(Mh_ Gamyy);
	if(Mh_ Gamyz) cudaFree(Mh_ Gamyz);
	if(Mh_ Gamzz) cudaFree(Mh_ Gamzz);
	if(Mh_ Gamyx) cudaFree(Mh_ Gamyx);
	if(Mh_ Gamzy) cudaFree(Mh_ Gamzy);
	if(Mh_ Gamzx) cudaFree(Mh_ Gamzx);
	if(Mh_ div_beta) cudaFree(Mh_ div_beta);
	if(Mh_ S) cudaFree(Mh_ S);
	if(Mh_ f) cudaFree(Mh_ f);
	if(Mh_ fxx) cudaFree(Mh_ fxx);
	if(Mh_ fxy) cudaFree(Mh_ fxy);
	if(Mh_ fxz) cudaFree(Mh_ fxz);
	if(Mh_ fyy) cudaFree(Mh_ fyy);
	if(Mh_ fyz) cudaFree(Mh_ fyz);
	if(Mh_ fzz) cudaFree(Mh_ fzz);
	if(Mh_ gupxx) cudaFree(Mh_ gupxx);
	if(Mh_ gupxy) cudaFree(Mh_ gupxy);
	if(Mh_ gupxz) cudaFree(Mh_ gupxz);
	if(Mh_ gupyy) cudaFree(Mh_ gupyy);
	if(Mh_ gupyz) cudaFree(Mh_ gupyz);
	if(Mh_ gupzz) cudaFree(Mh_ gupzz);
	if(Mh_ Gamxa) cudaFree(Mh_ Gamxa);
	if(Mh_ Gamya) cudaFree(Mh_ Gamya);
	if(Mh_ Gamza) cudaFree(Mh_ Gamza);
	if(Mh_ alpn1) cudaFree(Mh_ alpn1);
	if(Mh_ chin1) cudaFree(Mh_ chin1);
	if(Mh_ fh) cudaFree(Mh_ fh);
	if(Mh_ fh2) cudaFree(Mh_ fh2);
	if(Mh_ gxx_rhs) cudaFree(Mh_ gxx_rhs);
	if(Mh_ gyy_rhs) cudaFree(Mh_ gyy_rhs);
	if(Mh_ gzz_rhs) cudaFree(Mh_ gzz_rhs);
	if (meta) free(meta);
	meta = NULL;
}

void gpu_init_meta(int * ex) {
	cudaSetDevice(DEVICE_ID);

	int matrix_size = ex[0] * ex[1] * ex[2];
	// Meta met;
	// Meta * meta = &met;
	if (!meta) {
		meta = (Meta *)malloc(sizeof(Meta));
	}

	cudaMalloc((void**)&(Mh_ X), ex[0] * sizeof(double));
	cudaMalloc((void**)&(Mh_ Y), ex[1] * sizeof(double));
	cudaMalloc((void**)&(Mh_ Z), ex[2] * sizeof(double));
	cudaMalloc((void**)&(Mh_ chi), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ trK), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Axx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Axy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Axz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ayz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ayy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Azz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Lap), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betax), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betay), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betaz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ chi_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ trK_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxx_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxy_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ gyy_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyz_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ gzz_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ Axx_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ Axy_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ Axz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ayz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ayy_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Azz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamx_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamy_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Lap_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betax_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betay_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betaz_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfx_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfy_rhs), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ dtSfz_rhs), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ rho), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Sxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Syz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Syy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Szz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxzz), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ Gamyxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Rxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Rxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Rxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ryy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ryz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Rzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ ham_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ movx_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ movy_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ movz_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gmx_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gmy_Res), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gmz_Res), matrix_size * sizeof(double));


	//1.2 local Data
	cudaMalloc((void**)&(Mh_ gxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ chix), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ chiy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ chiz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxyx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxzx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyyx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyzx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gzzx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxzy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyzy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gzzy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gxzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gyzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gzzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Lapx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Lapy), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ Lapz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betaxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betaxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betaxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betayy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betayz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betazz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betayx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betazy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ betazx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Kx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Ky), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Kz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamyx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamzx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ div_beta), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ S), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ f), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fxy), matrix_size * sizeof(double));

	cudaMalloc((void**)&(Mh_ fxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupxx), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupxy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupxz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupyy), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupyz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ gupzz), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamxa), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamya), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ Gamza), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ alpn1), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ chin1), matrix_size * sizeof(double));
	cudaMalloc((void**)&(Mh_ fh), (ex[0]+2)*(ex[1]+2)*(ex[2]+2) * sizeof(double));
	cudaMalloc((void**)&(Mh_ fh2), (ex[0]+3)*(ex[1]+3)*(ex[2]+3) * sizeof(double));

	//init local var
	cudaMemset(Mh_ gxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ chix,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ chiy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ chiz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxyx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxzx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyyx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyzx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gzzx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxzy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyzy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gzzy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxyz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gxzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyyz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gyzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gzzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Lapx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Lapy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Lapz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betaxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betaxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betaxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betayy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betayz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betazz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betayx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betazy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ betazx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Kx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Ky,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Kz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamyz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamyx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamzy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamzx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ div_beta,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ S,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ f,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fyz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ fzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupyy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupyz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ gupzz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamxa,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamya,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Gamza,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ alpn1,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ chin1,0,matrix_size * sizeof(double));
}

void gpu_to_device( 
	int *ex, double *X, double *Y, double *Z, double *chi, double *  trK ,                                             
	double *dxx , double *  gxy    ,double *gxz ,double * dyy,double *gyz,double *dzz,     
	double *Axx ,   double *Axy ,  double * Axz ,  double * Ayy ,  double * Ayz , double * Azz,     
	double *Gamx ,  double *Gamy ,  double *Gamz ,                                  
	double *Lap ,  double *betax ,  double *betay ,  double *betaz ,                       
	double *dtSfx,  double *dtSfy ,  double *dtSfz
) {
	int matrix_size = ex[0] * ex[1] * ex[2];
	cudaMemcpy(Mh_ X, X, ex[0] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Y, Y, ex[1] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Z, Z, ex[2] * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ chi, chi, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dxx, dxx, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dyy, dyy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dzz, dzz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ trK, trK, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ gxy, gxy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ gxz, gxz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ gyz, gyz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Axx, Axx, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Axy, Axy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Axz, Axz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Ayz, Ayz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Ayy, Ayy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Azz, Azz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Gamx, Gamx, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Gamy, Gamy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Gamz, Gamz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ betax, betax, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ betay, betay, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ betaz, betaz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ Lap, Lap, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dtSfx, dtSfx, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dtSfy, dtSfy, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Mh_ dtSfz, dtSfz, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(Mh_ rho,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sxx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sxy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sxz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Syz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Syy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Szz,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sx,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sy,0,matrix_size * sizeof(double));
	cudaMemset(Mh_ Sz,0,matrix_size * sizeof(double));
}

void gpu_init_constant(
	int *ex, double &T, double *X, double *Y, double *Z,                                     
	int & Symmetry,int &Lev, double &eps, int &co
) {
	cudaMemcpyToSymbol(metac,meta, sizeof(Meta));
	cudaMemcpyToSymbol(ex_c,ex, 3*sizeof(int));
	cudaMemcpyToSymbol(T_c,&T, sizeof(double));
	cudaMemcpyToSymbol(Symmetry_c,&Symmetry, sizeof(int));
	cudaMemcpyToSymbol(Lev_c,&Lev, sizeof(int));
	cudaMemcpyToSymbol(co_c,&co, sizeof(int));
	cudaMemcpyToSymbol(eps_c,&eps, sizeof(double));
	
	double F1o3h  = 1.0;	F1o3h /= 3.0;
	double F2o3h  = 2.0;	F2o3h /= 3.0;
	double F1o6h  = 1.0;	F1o6h /= 6.0;
	double PIh = M_PI;
	int step = GRID_DIM * BLOCK_DIM;
	double dXh = X[1] - X[0];
	double dYh = Y[1] - Y[0];
	double dZh = Z[1] - Z[0];
	
	cudaMemcpyToSymbol(F1o3,&F1o3h, sizeof(double));
	cudaMemcpyToSymbol(F2o3,&F2o3h, sizeof(double));
	cudaMemcpyToSymbol(F1o6,&F1o6h, sizeof(double));
	cudaMemcpyToSymbol(PI,&PIh, sizeof(double));
	cudaMemcpyToSymbol(STEP_SIZE,&step, sizeof(int));
	cudaMemcpyToSymbol(dX,&dXh, sizeof(double));
	cudaMemcpyToSymbol(dY,&dYh, sizeof(double));
	cudaMemcpyToSymbol(dZ,&dZh, sizeof(double));
	
	int _1d_size[4];
	int _2d_size[4];
	int _3d_size[4];
	for(int i = 0;i<4;++i){
		_1d_size[i] = ex[0] + i;
		_2d_size[i] = _1d_size[i] * (ex[1]+i);
		_3d_size[i] = _2d_size[i] * (ex[2]+i);
	}
	cudaMemcpyToSymbol(_1D_SIZE,_1d_size, 4*sizeof(int));
	cudaMemcpyToSymbol(_2D_SIZE,_2d_size, 4*sizeof(int));
	cudaMemcpyToSymbol(_3D_SIZE,_3d_size, 4*sizeof(int));

	
//3.2--------for fderivs------------
	int ijkmax_h[3] = {ex[0]-1,ex[1]-1,ex[2]-1};
	int ijkmin_h[3] = {0,0,0};
	int ijkmin2_h[3] = {0,0,0};
	int ijkmin3_h[3] = {0,0,0};
	
	double abs[3] = {X[0],Y[0],Z[0]};
	for(int i = 0;i<3;++i){
		if(abs[i] < 0) abs[i] = -abs[i]; 
	}
  	if(Symmetry > 1 && abs[0] < dXh) {ijkmin_h[0] = -2; ijkmin2_h[0] = -3;}
  	if(Symmetry > 1 && abs[1] < dYh) {ijkmin_h[1] = -2; ijkmin2_h[1] = -3;}
  	if(Symmetry > 0 && abs[2] < dZh) {ijkmin_h[2] = -2; ijkmin2_h[2] = -3;}
  	
  	if(Symmetry > 2 && abs[0] < dXh) {ijkmin3_h[0] = -3;}
  	if(Symmetry > 2 && abs[1] < dYh) {ijkmin3_h[1] = -3;}
  	if(Symmetry > 0 && abs[2] < dZh) {ijkmin3_h[2] = -3;}
  	
  	cudaMemcpyToSymbol(ijk_max,ijkmax_h,3*sizeof(int));
  	cudaMemcpyToSymbol(ijk_min,ijkmin_h,3*sizeof(int));
  	cudaMemcpyToSymbol(ijk_min2,ijkmin2_h,3*sizeof(int));
  	cudaMemcpyToSymbol(ijk_min3,ijkmin3_h,3*sizeof(int));
	
	double d12dxyz_h[3] = {1.0,1.0,1.0};
	double d2dxyz_h[3] = {1.0,1.0,1.0};
	d12dxyz_h[0] /= 12; d12dxyz_h[1] /= 12; d12dxyz_h[2] /= 12;
	d12dxyz_h[0] /= dXh; d12dxyz_h[1] /= dYh; d12dxyz_h[2] /= dZh;
	d2dxyz_h[0] /= 2; d2dxyz_h[1] /= 2; d2dxyz_h[2] /= 2;
	d2dxyz_h[0] /= dXh; d2dxyz_h[1] /= dYh; d2dxyz_h[2] /= dZh;
	
	cudaMemcpyToSymbol(d12dxyz,d12dxyz_h,3*sizeof(double));
	cudaMemcpyToSymbol(d2dxyz,d2dxyz_h,3*sizeof(double));
	
//3.3--------for fdderivs------------
	double Sdxdxh =  1.0 /( dXh * dXh ); 
	double Sdydyh =  1.0 /( dYh * dYh );
	double Sdzdzh =  1.0 /( dZh * dZh );
	double Fdxdxh = 1.0 / 12.0 /( dXh * dXh );
	double Fdydyh = 1.0 / 12.0 /( dYh * dYh );
	double Fdzdzh = 1.0 / 12.0 /( dZh * dZh );
	double Sdxdyh = 1.0/4.0 /( dXh * dYh );
	double Sdxdzh = 1.0/4.0 /( dXh * dZh );
	double Sdydzh = 1.0/4.0 /( dYh * dZh );
	double Fdxdyh = 1.0/144.0 /( dXh * dYh );
	double Fdxdzh = 1.0/144.0 /( dXh * dZh );
	double Fdydzh = 1.0/144.0 /( dYh * dZh );
	cudaMemcpyToSymbol(Sdxdx,&Sdxdxh,sizeof(double));
	cudaMemcpyToSymbol(Sdydy,&Sdydyh,sizeof(double));
	cudaMemcpyToSymbol(Sdzdz,&Sdzdzh,sizeof(double));
	cudaMemcpyToSymbol(Sdxdy,&Sdxdyh,sizeof(double));
	cudaMemcpyToSymbol(Sdxdz,&Sdxdzh,sizeof(double));
	cudaMemcpyToSymbol(Sdydz,&Sdydzh,sizeof(double));
	cudaMemcpyToSymbol(Fdxdx,&Fdxdxh,sizeof(double));
	cudaMemcpyToSymbol(Fdydy,&Fdydyh,sizeof(double));
	cudaMemcpyToSymbol(Fdzdz,&Fdzdzh,sizeof(double));
	cudaMemcpyToSymbol(Fdxdy,&Fdxdyh,sizeof(double));
	cudaMemcpyToSymbol(Fdxdz,&Fdxdzh,sizeof(double));
	cudaMemcpyToSymbol(Fdydz,&Fdydzh,sizeof(double));
}

void gpu_back_to_host(
	int calledby, int *ex, 
	double *chi_rhs, double *  trK_rhs,                                             
	double *gxx_rhs,  double * gxy_rhs,  double * gxz_rhs,  double * gyy_rhs,   double *gyz_rhs,  double * gzz_rhs, 
	double *Axx_rhs, double *  Axy_rhs,  double * Axz_rhs,  double * Ayy_rhs,   double *Ayz_rhs,   double *Azz_rhs, 
	double *Gamx_rhs, double * Gamy_rhs, double * Gamz_rhs,                                 
	double *Lap_rhs,  double *betax_rhs, double * betay_rhs, double * betaz_rhs,                    
	double *dtSfx_rhs, double * dtSfy_rhs, double * dtSfz_rhs,                          
	double *Gamxxx,double *Gamxxy,double *Gamxxz,double *Gamxyy,double *Gamxyz,double *Gamxzz,                      
	double *Gamyxx,double *Gamyxy,double *Gamyxz,double *Gamyyy,double *Gamyyz,double *Gamyzz,                      
	double *Gamzxx,double *Gamzxy,double *Gamzxz,double *Gamzyy,double *Gamzyz,double *Gamzzz,                      
	double *Rxx,double *Rxy,double *Rxz,double *Ryy,double *Ryz,double *Rzz,                                        
	double *ham_Res, double *movx_Res, double *movy_Res,double * movz_Res, 
	double * Gmx_Res, double *Gmy_Res,double * Gmz_Res
) {
	int matrix_size = ex[0] * ex[1] * ex[2];
	if(calledby == CALLED_BY_STEP) {	
		cudaMemcpy(chi_rhs, Mh_ chi_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(trK_rhs, Mh_ trK_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gxx_rhs, Mh_ gxx_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gxy_rhs, Mh_ gxy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gxz_rhs, Mh_ gxz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gyy_rhs, Mh_ gyy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gyz_rhs, Mh_ gyz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(gzz_rhs, Mh_ gzz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Axx_rhs, Mh_ Axx_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Axy_rhs, Mh_ Axy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Axz_rhs, Mh_ Axz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ayy_rhs, Mh_ Ayy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ayz_rhs, Mh_ Ayz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Azz_rhs, Mh_ Azz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamx_rhs, Mh_ Gamx_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamy_rhs, Mh_ Gamy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamz_rhs, Mh_ Gamz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Lap_rhs, Mh_ Lap_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(betax_rhs, Mh_ betax_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(betay_rhs, Mh_ betay_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(betaz_rhs, Mh_ betaz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(dtSfx_rhs, Mh_ dtSfx_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(dtSfy_rhs, Mh_ dtSfy_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(dtSfz_rhs, Mh_ dtSfz_rhs, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
	}
	else if(calledby == CALLED_BY_CONSTRAINT)
	{
		cudaMemcpy(Gamxxx, Mh_ Gamxxx, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamxxy, Mh_ Gamxxy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamxxz, Mh_ Gamxxz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamxyy, Mh_ Gamxyy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamxyz, Mh_ Gamxyz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamxzz, Mh_ Gamxzz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyxx, Mh_ Gamyxx, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyxy, Mh_ Gamyxy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyxz, Mh_ Gamyxz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyyy, Mh_ Gamyyy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyyz, Mh_ Gamyyz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamyzz, Mh_ Gamyzz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzxx, Mh_ Gamzxx, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzxy, Mh_ Gamzxy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzxz, Mh_ Gamzxz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzyy, Mh_ Gamzyy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzyz, Mh_ Gamzyz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gamzzz, Mh_ Gamzzz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Rxx, Mh_ Rxx, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Rxy, Mh_ Rxy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Rxz, Mh_ Rxz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ryy, Mh_ Ryy, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ryz, Mh_ Ryz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Rzz, Mh_ Rzz, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(ham_Res, Mh_ ham_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(movx_Res, Mh_ movx_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(movy_Res, Mh_ movy_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(movz_Res, Mh_ movz_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gmx_Res, Mh_ Gmx_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gmy_Res, Mh_ Gmy_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Gmz_Res, Mh_ Gmz_Res, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
	}
}