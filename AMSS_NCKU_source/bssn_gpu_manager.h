#ifndef BSSN_GPU_MANAGER_H
#define BSSN_GPU_MANAGER_H

#include "gpu_mem.h"


struct GPU_CALL_CONTEXT {
	int calledby, mpi_rank, *ex;
	double &T, *X, *Y, *Z;                                    
	double *chi, *trK;                                            
	double *dxx , *gxy    , *gxz , * dyy, *gyz, *dzz;  
	double *Axx ,    *Axy ,   * Axz ,   * Ayy ,   * Ayz ,  * Azz;     
	double *Gamx ,   *Gamy ,   *Gamz;                                  
	double *Lap ,   *betax ,   *betay ,   *betaz;                       
	double *dtSfx,   *dtSfy ,   *dtSfz;                                  
	double *chi_rhs,  *  trK_rhs;                                             
	double *gxx_rhs,   * gxy_rhs,   * gxz_rhs,   * gyy_rhs,    *gyz_rhs,   * gzz_rhs; 
	double *Axx_rhs,  *  Axy_rhs,   * Axz_rhs,   * Ayy_rhs,    *Ayz_rhs,    *Azz_rhs; 
	double *Gamx_rhs,  * Gamy_rhs,  * Gamz_rhs;                                 
	double *Lap_rhs,   *betax_rhs,  * betay_rhs,  * betaz_rhs;                    
	double *dtSfx_rhs,  * dtSfy_rhs,  * dtSfz_rhs;                              
	double *rho, *Sx, *Sy, *Sz, *Sxx;
	double *Sxy, *Sxz, *Syy, *Syz, *Szz;                           
	double *Gamxxx, *Gamxxy, *Gamxxz, *Gamxyy, *Gamxyz, *Gamxzz;                      
	double *Gamyxx, *Gamyxy, *Gamyxz, *Gamyyy, *Gamyyz, *Gamyzz;                      
	double *Gamzxx, *Gamzxy, *Gamzxz, *Gamzyy, *Gamzyz, *Gamzzz;                      
	double *Rxx, *Rxy, *Rxz, *Ryy, *Ryz, *Rzz;                                  
	double *ham_Res,  *movx_Res,  *movy_Res, * movz_Res;
	double * Gmx_Res,  *Gmy_Res, * Gmz_Res;
	int & Symmetry, &Lev; double &eps; int &co;
};

Meta * gpu_get_meta();

void gpu_destroy_meta();

// void gpu_init_meta(int * ex);
void gpu_init_meta(GPU_CALL_CONTEXT &ctx);

// void gpu_to_device( 
// 	int *ex, double *X, double *Y, double *Z, double *chi, double *  trK ,                                             
// 	double *dxx , double *  gxy    ,double *gxz ,double * dyy,double *gyz,double *dzz,     
// 	double *Axx ,   double *Axy ,  double * Axz ,  double * Ayy ,  double * Ayz , double * Azz,     
// 	double *Gamx ,  double *Gamy ,  double *Gamz ,                                  
// 	double *Lap ,  double *betax ,  double *betay ,  double *betaz ,                       
// 	double *dtSfx,  double *dtSfy ,  double *dtSfz
// );
void gpu_to_device(GPU_CALL_CONTEXT &ctx);

// void gpu_init_constant(
// 	int *ex, double &T, double *X, double *Y, double *Z,                                     
// 	int & Symmetry,int &Lev, double &eps, int &co
// );
void gpu_init_constant(GPU_CALL_CONTEXT &ctx);

// void gpu_back_to_host(
// 	int calledby, int *ex, 
// 	double *chi_rhs, double *  trK_rhs,                                             
// 	double *gxx_rhs,  double * gxy_rhs,  double * gxz_rhs,  double * gyy_rhs,   double *gyz_rhs,  double * gzz_rhs, 
// 	double *Axx_rhs, double *  Axy_rhs,  double * Axz_rhs,  double * Ayy_rhs,   double *Ayz_rhs,   double *Azz_rhs, 
// 	double *Gamx_rhs, double * Gamy_rhs, double * Gamz_rhs,                                 
// 	double *Lap_rhs,  double *betax_rhs, double * betay_rhs, double * betaz_rhs,                    
// 	double *dtSfx_rhs, double * dtSfy_rhs, double * dtSfz_rhs,                          
// 	double *Gamxxx,double *Gamxxy,double *Gamxxz,double *Gamxyy,double *Gamxyz,double *Gamxzz,                      
// 	double *Gamyxx,double *Gamyxy,double *Gamyxz,double *Gamyyy,double *Gamyyz,double *Gamyzz,                      
// 	double *Gamzxx,double *Gamzxy,double *Gamzxz,double *Gamzyy,double *Gamzyz,double *Gamzzz,                      
// 	double *Rxx,double *Rxy,double *Rxz,double *Ryy,double *Ryz,double *Rzz,                                        
// 	double *ham_Res, double *movx_Res, double *movy_Res,double * movz_Res, 
// 	double * Gmx_Res, double *Gmy_Res,double * Gmz_Res
// );
void gpu_back_to_host(GPU_CALL_CONTEXT &ctx);

#endif