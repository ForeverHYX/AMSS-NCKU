// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sys/time.h>

#ifdef RESULT_CHECK
#include <fstream>
#endif

// include BSSN class files
#include "macrodef.h"
#include "fmisc.h"
#include "bssn_gpu_class.h"
#include "bssn_rhs.h"
#include "enforce_algebra.h"
#include "rungekutta4_rout.h"
#include "sommerfeld_rout.h"

// include gpu files
#include "bssn_gpu_manager.h"

void bssn_class::Step_GPU(int lev, int YN)
{
	setpbh(BH_num, Porg0, Mass, BH_num_input);

	double dT_lev = dT * pow(0.5, Mymax(lev, trfls));

// new code 2013-2-15, zjcao
	// for black hole position
	if (BH_num > 0 && lev == GH->levels - 1)
	{
		compute_Porg_rhs(Porg0, Porg_rhs, Sfx0, Sfy0, Sfz0, lev);
		for (int ithBH = 0; ithBH < BH_num; ithBH++)
		{
			for (int ith = 0; ith < 3; ith++)
				Porg1[ithBH][ith] = Porg0[ithBH][ith] + Porg_rhs[ithBH][ith] * dT_lev;
			if (Symmetry > 0)
				Porg1[ithBH][2] = fabs(Porg1[ithBH][2]);
			if (Symmetry == 2)
			{
				Porg1[ithBH][0] = fabs(Porg1[ithBH][0]);
				Porg1[ithBH][1] = fabs(Porg1[ithBH][1]);
			}
			if (!finite(Porg1[ithBH][0]) || !finite(Porg1[ithBH][1]) || !finite(Porg1[ithBH][2]))
			{
				if (ErrorMonitor->outfile)
					ErrorMonitor->outfile << "predictor step finds NaN for BH's position from ("
																<< Porg0[ithBH][0] << "," << Porg0[ithBH][1] << "," << Porg0[ithBH][2] << ")" << endl;

				MyList<var> *DG_List = new MyList<var>(Sfx0);
				DG_List->insert(Sfx0);
				DG_List->insert(Sfy0);
				DG_List->insert(Sfz0);
				Parallel::Dump_Data(GH->PatL[lev], DG_List, 0, PhysTime, dT_lev);
				DG_List->clearList();
			}
		}
	}

	// data analysis part
	// Warning NOTE: the variables1 are used as temp storege room
	if (lev == a_lev)
	{
		AnalysisStuff(lev, dT_lev);
	}

	bool BB = fgt(PhysTime, StartTime, dT_lev / 2);
	double ndeps = numepss;
	if (lev < GH->movls)
		ndeps = numepsb;
	double TRK4 = PhysTime;
	int iter_count = 0; // count RK4 substeps
	int pre = 0, cor = 1;
	int ERROR = 0;

	MyList<ss_patch> *sPp;
	// Predictor
	MyList<Patch> *Pp = GH->PatL[lev];
	while (Pp)
	{
		MyList<Block> *BP = Pp->data->blb;
		while (BP)
		{
			Block *cg = BP->data;
			if (myrank == cg->rank)
			{
				f_enforce_ga(
					cg->shape,
					cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
					cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn]
				);

				if (gpu_compute_rhs_bssn(
					cg->shape, TRK4, cg->X[0], cg->X[1], cg->X[2],
					cg->fgfs[phi0->sgfn], cg->fgfs[trK0->sgfn],
					cg->fgfs[gxx0->sgfn], cg->fgfs[gxy0->sgfn], cg->fgfs[gxz0->sgfn], 
					cg->fgfs[gyy0->sgfn], cg->fgfs[gyz0->sgfn], cg->fgfs[gzz0->sgfn],
					cg->fgfs[Axx0->sgfn], cg->fgfs[Axy0->sgfn], cg->fgfs[Axz0->sgfn], 
					cg->fgfs[Ayy0->sgfn], cg->fgfs[Ayz0->sgfn], cg->fgfs[Azz0->sgfn],
					cg->fgfs[Gmx0->sgfn], cg->fgfs[Gmy0->sgfn], cg->fgfs[Gmz0->sgfn],
					cg->fgfs[Lap0->sgfn], 
					cg->fgfs[Sfx0->sgfn], cg->fgfs[Sfy0->sgfn], cg->fgfs[Sfz0->sgfn],
					cg->fgfs[dtSfx0->sgfn], cg->fgfs[dtSfy0->sgfn], cg->fgfs[dtSfz0->sgfn],
					cg->fgfs[phi_rhs->sgfn], cg->fgfs[trK_rhs->sgfn],
					cg->fgfs[gxx_rhs->sgfn], cg->fgfs[gxy_rhs->sgfn], cg->fgfs[gxz_rhs->sgfn],
					cg->fgfs[gyy_rhs->sgfn], cg->fgfs[gyz_rhs->sgfn], cg->fgfs[gzz_rhs->sgfn],
					cg->fgfs[Axx_rhs->sgfn], cg->fgfs[Axy_rhs->sgfn], cg->fgfs[Axz_rhs->sgfn],
					cg->fgfs[Ayy_rhs->sgfn], cg->fgfs[Ayz_rhs->sgfn], cg->fgfs[Azz_rhs->sgfn],
					cg->fgfs[Gmx_rhs->sgfn], cg->fgfs[Gmy_rhs->sgfn], cg->fgfs[Gmz_rhs->sgfn],
					cg->fgfs[Lap_rhs->sgfn], 
					cg->fgfs[Sfx_rhs->sgfn], cg->fgfs[Sfy_rhs->sgfn], cg->fgfs[Sfz_rhs->sgfn],
					cg->fgfs[dtSfx_rhs->sgfn], cg->fgfs[dtSfy_rhs->sgfn], cg->fgfs[dtSfz_rhs->sgfn],
					cg->fgfs[rho->sgfn], cg->fgfs[Sx->sgfn], cg->fgfs[Sy->sgfn], cg->fgfs[Sz->sgfn],
					cg->fgfs[Sxx->sgfn], cg->fgfs[Sxy->sgfn], cg->fgfs[Sxz->sgfn], 
					cg->fgfs[Syy->sgfn], cg->fgfs[Syz->sgfn], cg->fgfs[Szz->sgfn],
					cg->fgfs[Gamxxx->sgfn], cg->fgfs[Gamxxy->sgfn], cg->fgfs[Gamxxz->sgfn],
					cg->fgfs[Gamxyy->sgfn], cg->fgfs[Gamxyz->sgfn], cg->fgfs[Gamxzz->sgfn],
					cg->fgfs[Gamyxx->sgfn], cg->fgfs[Gamyxy->sgfn], cg->fgfs[Gamyxz->sgfn],
					cg->fgfs[Gamyyy->sgfn], cg->fgfs[Gamyyz->sgfn], cg->fgfs[Gamyzz->sgfn],
					cg->fgfs[Gamzxx->sgfn], cg->fgfs[Gamzxy->sgfn], cg->fgfs[Gamzxz->sgfn],
					cg->fgfs[Gamzyy->sgfn], cg->fgfs[Gamzyz->sgfn], cg->fgfs[Gamzzz->sgfn],
					cg->fgfs[Rxx->sgfn], cg->fgfs[Rxy->sgfn], cg->fgfs[Rxz->sgfn], 
					cg->fgfs[Ryy->sgfn], cg->fgfs[Ryz->sgfn], cg->fgfs[Rzz->sgfn],
					cg->fgfs[Cons_Ham->sgfn],
					cg->fgfs[Cons_Px->sgfn], cg->fgfs[Cons_Py->sgfn], cg->fgfs[Cons_Pz->sgfn],
					cg->fgfs[Cons_Gx->sgfn], cg->fgfs[Cons_Gy->sgfn], cg->fgfs[Cons_Gz->sgfn],
					Symmetry, lev, ndeps, pre
				)) {
					cout << "find NaN in domain: (" << cg->bbox[0] << ":" << cg->bbox[3] << "," << cg->bbox[1] << ":" << cg->bbox[4] << ","
							 << cg->bbox[2] << ":" << cg->bbox[5] << ")" << endl;
					ERROR = 1;
				}
				// rk4 substep and boundary
				{
					MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varlrhs = RHSList; // we do not check the correspondence here
					while (varl0)
					{
						if (lev == 0) // sommerfeld indeed
							f_sommerfeld_routbam(cg->shape, cg->X[0], cg->X[1], cg->X[2],
																	 Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
																	 cg->fgfs[varlrhs->data->sgfn],
																	 cg->fgfs[varl0->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
																	 Symmetry);
						f_rungekutta4_rout(cg->shape, dT_lev, cg->fgfs[varl0->data->sgfn], cg->fgfs[varl->data->sgfn], cg->fgfs[varlrhs->data->sgfn],
															 iter_count);
						if (lev > 0) // fix BD point
							f_sommerfeld_rout(cg->shape, cg->X[0], cg->X[1], cg->X[2],
																Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
																dT_lev, cg->fgfs[phi0->sgfn],
																cg->fgfs[Lap0->sgfn], cg->fgfs[varl0->data->sgfn], cg->fgfs[varl->data->sgfn], varl0->data->SoA,
																Symmetry, cor);

						varl0 = varl0->next;
						varl = varl->next;
						varlrhs = varlrhs->next;
					}
				}
				f_lowerboundset(cg->shape, cg->fgfs[phi->sgfn], chitiny);
			}
			if (BP == Pp->data->ble)
				break;
			BP = BP->next;
		}
		Pp = Pp->next;
	}
	// check error information
	{
		int erh = ERROR;
		MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}
	if (ERROR)
	{
		Parallel::Dump_Data(GH->PatL[lev], StateList, 0, PhysTime, dT_lev);
		if (myrank == 0)
		{
			if (ErrorMonitor->outfile)
				ErrorMonitor->outfile << "find NaN in state variables at t = " << PhysTime << ", lev = " << lev << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	Parallel::Sync(GH->PatL[lev], SynchList_pre, Symmetry);

	// corrector
	for (iter_count = 1; iter_count < 4; iter_count++)
	{
		// for RK4: t0, t0+dt/2, t0+dt/2, t0+dt;
		if (iter_count == 1 || iter_count == 3)
			TRK4 += dT_lev / 2;
		Pp = GH->PatL[lev];
		while (Pp)
		{
			MyList<Block> *BP = Pp->data->blb;
			while (BP)
			{
				Block *cg = BP->data;
				if (myrank == cg->rank)
				{
						f_enforce_ga(
							cg->shape,
							cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
							cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn]
						);
						
						if (gpu_compute_rhs_bssn(
							cg->shape, TRK4, cg->X[0], cg->X[1], cg->X[2],
							cg->fgfs[phi->sgfn], cg->fgfs[trK->sgfn],
							cg->fgfs[gxx->sgfn], cg->fgfs[gxy->sgfn], cg->fgfs[gxz->sgfn], 
							cg->fgfs[gyy->sgfn], cg->fgfs[gyz->sgfn], cg->fgfs[gzz->sgfn],
							cg->fgfs[Axx->sgfn], cg->fgfs[Axy->sgfn], cg->fgfs[Axz->sgfn], 
							cg->fgfs[Ayy->sgfn], cg->fgfs[Ayz->sgfn], cg->fgfs[Azz->sgfn],
							cg->fgfs[Gmx->sgfn], cg->fgfs[Gmy->sgfn], cg->fgfs[Gmz->sgfn],
							cg->fgfs[Lap->sgfn], 
							cg->fgfs[Sfx->sgfn], cg->fgfs[Sfy->sgfn], cg->fgfs[Sfz->sgfn],
							cg->fgfs[dtSfx->sgfn], cg->fgfs[dtSfy->sgfn], cg->fgfs[dtSfz->sgfn],
							cg->fgfs[phi1->sgfn], cg->fgfs[trK1->sgfn],
							cg->fgfs[gxx1->sgfn], cg->fgfs[gxy1->sgfn], cg->fgfs[gxz1->sgfn],
							cg->fgfs[gyy1->sgfn], cg->fgfs[gyz1->sgfn], cg->fgfs[gzz1->sgfn],
							cg->fgfs[Axx1->sgfn], cg->fgfs[Axy1->sgfn], cg->fgfs[Axz1->sgfn],
							cg->fgfs[Ayy1->sgfn], cg->fgfs[Ayz1->sgfn], cg->fgfs[Azz1->sgfn],
							cg->fgfs[Gmx1->sgfn], cg->fgfs[Gmy1->sgfn], cg->fgfs[Gmz1->sgfn],
							cg->fgfs[Lap1->sgfn], 
							cg->fgfs[Sfx1->sgfn], cg->fgfs[Sfy1->sgfn], cg->fgfs[Sfz1->sgfn],
							cg->fgfs[dtSfx1->sgfn], cg->fgfs[dtSfy1->sgfn], cg->fgfs[dtSfz1->sgfn],
							cg->fgfs[rho->sgfn], 
							cg->fgfs[Sx->sgfn], cg->fgfs[Sy->sgfn], cg->fgfs[Sz->sgfn],
							cg->fgfs[Sxx->sgfn], cg->fgfs[Sxy->sgfn], cg->fgfs[Sxz->sgfn], 
							cg->fgfs[Syy->sgfn], cg->fgfs[Syz->sgfn], cg->fgfs[Szz->sgfn],
							cg->fgfs[Gamxxx->sgfn], cg->fgfs[Gamxxy->sgfn], cg->fgfs[Gamxxz->sgfn],
							cg->fgfs[Gamxyy->sgfn], cg->fgfs[Gamxyz->sgfn], cg->fgfs[Gamxzz->sgfn],
							cg->fgfs[Gamyxx->sgfn], cg->fgfs[Gamyxy->sgfn], cg->fgfs[Gamyxz->sgfn],
							cg->fgfs[Gamyyy->sgfn], cg->fgfs[Gamyyz->sgfn], cg->fgfs[Gamyzz->sgfn],
							cg->fgfs[Gamzxx->sgfn], cg->fgfs[Gamzxy->sgfn], cg->fgfs[Gamzxz->sgfn],
							cg->fgfs[Gamzyy->sgfn], cg->fgfs[Gamzyz->sgfn], cg->fgfs[Gamzzz->sgfn],
							cg->fgfs[Rxx->sgfn], cg->fgfs[Rxy->sgfn], cg->fgfs[Rxz->sgfn], 
							cg->fgfs[Ryy->sgfn], cg->fgfs[Ryz->sgfn], cg->fgfs[Rzz->sgfn],
							cg->fgfs[Cons_Ham->sgfn],
							cg->fgfs[Cons_Px->sgfn], cg->fgfs[Cons_Py->sgfn], cg->fgfs[Cons_Pz->sgfn],
							cg->fgfs[Cons_Gx->sgfn], cg->fgfs[Cons_Gy->sgfn], cg->fgfs[Cons_Gz->sgfn],
							Symmetry, lev, ndeps, cor)
						){
								cout << "find NaN in domain: (" << cg->bbox[0] << ":" << cg->bbox[3] << "," << cg->bbox[1] << ":" << cg->bbox[4] << ","
										<< cg->bbox[2] << ":" << cg->bbox[5] << ")" << endl;
								ERROR = 1;
						}
					// rk4 substep and boundary
					{
						MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varl1 = SynchList_cor, *varlrhs = RHSList; // we do not check the correspondence here
						while (varl0)
						{
							if (lev == 0) // sommerfeld indeed
								f_sommerfeld_routbam(cg->shape, cg->X[0], cg->X[1], cg->X[2],
																		 Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
																		 cg->fgfs[varl1->data->sgfn],
																		 cg->fgfs[varl->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
																		 Symmetry);
							f_rungekutta4_rout(cg->shape, dT_lev, cg->fgfs[varl0->data->sgfn], cg->fgfs[varl1->data->sgfn], cg->fgfs[varlrhs->data->sgfn],
																 iter_count);
							if (lev > 0) // fix BD point
								f_sommerfeld_rout(cg->shape, cg->X[0], cg->X[1], cg->X[2],
																	Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
																	dT_lev, cg->fgfs[phi0->sgfn],
																	cg->fgfs[Lap0->sgfn], cg->fgfs[varl0->data->sgfn], cg->fgfs[varl1->data->sgfn], varl0->data->SoA,
																	Symmetry, cor);

							varl0 = varl0->next;
							varl = varl->next;
							varl1 = varl1->next;
							varlrhs = varlrhs->next;
						}
					}
					f_lowerboundset(cg->shape, cg->fgfs[phi1->sgfn], chitiny);
				}
				if (BP == Pp->data->ble)
					break;
				BP = BP->next;
			}
			Pp = Pp->next;
		}

		// check error information
		{
			int erh = ERROR;
			MPI_Allreduce(&erh, &ERROR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		}

		if (ERROR)
		{
			Parallel::Dump_Data(GH->PatL[lev], SynchList_pre, 0, PhysTime, dT_lev);
			if (myrank == 0)
			{
				if (ErrorMonitor->outfile)
					ErrorMonitor->outfile << "find NaN in RK4 substep#" << iter_count << " variables at t = " << PhysTime << ", lev = " << lev << endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		Parallel::Sync(GH->PatL[lev], SynchList_cor, Symmetry);

		// swap time level
		if (iter_count < 3)
		{
			Pp = GH->PatL[lev];
			while (Pp)
			{
				MyList<Block> *BP = Pp->data->blb;
				while (BP)
				{
					Block *cg = BP->data;
					cg->swapList(SynchList_pre, SynchList_cor, myrank);
					if (BP == Pp->data->ble)
						break;
					BP = BP->next;
				}
				Pp = Pp->next;
			}
		}
	}
	// note the data structure before update
	// SynchList_cor 1   -----------
	//
	// StateList     0   -----------
	//
	// OldStateList  old -----------
	// update
	Pp = GH->PatL[lev];
	while (Pp)
	{
		MyList<Block> *BP = Pp->data->blb;
		while (BP)
		{
			Block *cg = BP->data;
			cg->swapList(StateList, SynchList_cor, myrank);
			cg->swapList(OldStateList, SynchList_cor, myrank);
			if (BP == Pp->data->ble)
				break;
			BP = BP->next;
		}
		Pp = Pp->next;
	}
	// for black hole position
	if (BH_num > 0 && lev == GH->levels - 1)
	{
		for (int ithBH = 0; ithBH < BH_num; ithBH++)
		{
			Porg0[ithBH][0] = Porg1[ithBH][0];
			Porg0[ithBH][1] = Porg1[ithBH][1];
			Porg0[ithBH][2] = Porg1[ithBH][2];
		}
	}
}
