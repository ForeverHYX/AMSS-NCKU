#include "Parallel.h"

#include "gpu_manager.h"
#include "prolongrestrict.h"
#include "misc.h"
#include "parameters.h"
#include "fmisc.h"

#include "prolongrestrict_gpu_manager.h"
#include "gpu_manager.h"


int Parallel::gpu_data_packer(
    double *d_data, MyList<Parallel::gridseg> *src, MyList<Parallel::gridseg> *dst, int rank_in, int dir,
    MyList<var> *VarLists, MyList<var> *VarListd, int Symmetry
) {
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int DIM = dim;

    if (dir != PACK && dir != UNPACK) {
        cout << "error dir " << dir << " for data_packer " << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int size_out = 0;

    if (!src || !dst) return size_out;

    MyList<var> *varls, *varld;

    varls = VarLists;
    varld = VarListd;
    while (varls && varld) {
        varls = varls->next;
        varld = varld->next;
    }

    if (varls || varld) {
        cout << "error in short data packer, var lists does not match." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int type; /* 1 copy, 2 restrict, 3 prolong */
    if (src->data->Bg->lev == dst->data->Bg->lev) type = 1;
    else if (src->data->Bg->lev > dst->data->Bg->lev) type = 2;
    else type = 3;
    
    while (src && dst) {
        if (
            (dir == PACK && dst->data->Bg->rank == rank_in && src->data->Bg->rank == myrank) ||
            (dir == UNPACK && src->data->Bg->rank == rank_in && dst->data->Bg->rank == myrank)
        ) {
            varls = VarLists;
            varld = VarListd;
            while (varls && varld) {
                if (d_data) {
                    if (dir == PACK) {
                        double* d_dst_ptr = d_data + size_out; 
                        double* d_src_ptr = src->data->Bg->d_fgfs[varls->data->sgfn];

                        switch (type) {
                        case 1: {
                            double dx = (src->data->Bg->bbox[3] - src->data->Bg->bbox[0]) / src->data->Bg->shape[0];
                            double dy = (src->data->Bg->bbox[4] - src->data->Bg->bbox[1]) / src->data->Bg->shape[1];
                            double dz = (src->data->Bg->bbox[5] - src->data->Bg->bbox[2]) / src->data->Bg->shape[2];

                            // 计算幽灵区在源大网格中的 0-based 起始索引偏移 (使用 std::trunc 对齐 Fortran 的 idint)
                            int off_x = (int)std::trunc((dst->data->llb[0] - src->data->Bg->bbox[0]) / dx + 0.4);
                            int off_y = (int)std::trunc((dst->data->llb[1] - src->data->Bg->bbox[1]) / dy + 0.4);
                            int off_z = (int)std::trunc((dst->data->llb[2] - src->data->Bg->bbox[2]) / dz + 0.4);

                            gpu_pack_launch(
                                src->data->Bg->stream, 
                                d_src_ptr, d_dst_ptr,
                                src->data->Bg->shape[0], src->data->Bg->shape[1], // 源 3D 数组的 XY 维度
                                dst->data->shape[0], dst->data->shape[1], dst->data->shape[2], // 目标幽灵区的大小
                                off_x, off_y, off_z
                            );
                            break;
                        }

                        case 2: {
                            gpu_restrict3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_f, dst_c
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry
                            );
                            break;
                        }

                        case 3: {
                            gpu_prolong3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_c, dst_f
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry
                            );
                            break;
                        }
                        }
                    }
                    
                    if (dir == UNPACK) {
                        double* d_src_ptr = d_data + size_out;
                        double* d_dst_ptr = dst->data->Bg->d_fgfs[varld->data->sgfn];

                        double dx = (dst->data->Bg->bbox[3] - dst->data->Bg->bbox[0]) / dst->data->Bg->shape[0];
                        double dy = (dst->data->Bg->bbox[4] - dst->data->Bg->bbox[1]) / dst->data->Bg->shape[1];
                        double dz = (dst->data->Bg->bbox[5] - dst->data->Bg->bbox[2]) / dst->data->Bg->shape[2];

                        // 计算收到的一维幽灵区数据在目标大网格中的 0-based 写入起始索引
                        int off_x = (int)std::trunc((dst->data->llb[0] - dst->data->Bg->bbox[0]) / dx + 0.4);
                        int off_y = (int)std::trunc((dst->data->llb[1] - dst->data->Bg->bbox[1]) / dy + 0.4);
                        int off_z = (int)std::trunc((dst->data->llb[2] - dst->data->Bg->bbox[2]) / dz + 0.4);

                        gpu_unpack_launch(
                            dst->data->Bg->stream, 
                            d_src_ptr, d_dst_ptr,
                            dst->data->Bg->shape[0], dst->data->Bg->shape[1], // 目标 3D 数组的 XY 维度
                            dst->data->shape[0], dst->data->shape[1], dst->data->shape[2], // 收到幽灵区数据的大小
                            off_x, off_y, off_z
                        );
                    }
                }
                size_out += dst->data->shape[0] * dst->data->shape[1] * dst->data->shape[2];
                varls = varls->next;
                varld = varld->next;
            }
        }
        dst = dst->next;
        src = src->next;
    }
    GPUManager::getInstance().synchronize_all();

    return size_out;
}

#if MPI_CUDA_AWARE
void Parallel::gpu_transfer(
    MyList<gridseg> **src, MyList<gridseg> **dst,
    MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /*target */,
    int Symmetry
) {
    int myrank, cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int node;

    MPI_Request *reqs;
    MPI_Status *stats;
    reqs = new MPI_Request[2 * cpusize];
    stats = new MPI_Status[2 * cpusize];
    int req_no = 0;

    double **send_data, **rec_data;
    send_data = new double *[cpusize];
    rec_data = new double *[cpusize];
    int length;
    for (node = 0; node < cpusize; node++) {
        send_data[node] = rec_data[node] = 0;
        if (node == myrank) {
            if (length = gpu_data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry)) {
                CUDA_CHECK(cudaMalloc((void**)&rec_data[node], length * sizeof(double)));
                gpu_data_packer(rec_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
            }
        }
        else {
            if (length = gpu_data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry)) {
                CUDA_CHECK(cudaMalloc((void**)&send_data[node], length * sizeof(double)));
                gpu_data_packer(send_data[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
                MPI_Isend((void *)send_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no ++);
            }
            if (length = gpu_data_packer(0, src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry)) {
                CUDA_CHECK(cudaMalloc((void**)&rec_data[node], length * sizeof(double)));
                MPI_Irecv((void *)rec_data[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no ++);
            }
        }
    }
    MPI_Waitall(req_no, reqs, stats);
    for (node = 0; node < cpusize; node++) {
        if (rec_data[node]) gpu_data_packer(rec_data[node], src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);
    }
        
    for (node = 0; node < cpusize; node++) {
        if (send_data[node]) CUDA_CHECK(cudaFree(send_data[node]));
        if (rec_data[node]) CUDA_CHECK(cudaFree(rec_data[node]));
    }

    delete[] reqs;
    delete[] stats;
    delete[] send_data;
    delete[] rec_data;
}
#else
void Parallel::gpu_transfer(
    MyList<gridseg> **src, MyList<gridseg> **dst,
    MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /*target */,
    int Symmetry
) {
    int myrank, cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int node;
    MPI_Request *reqs = new MPI_Request[2 * cpusize];
    MPI_Status *stats = new MPI_Status[2 * cpusize];
    int req_no = 0;

    // 分离 GPU 指针和 CPU 指针
    double **send_data_d = new double *[cpusize];
    double **rec_data_d  = new double *[cpusize];
    double **send_data_h = new double *[cpusize]; // CPU 发送暂存
    double **rec_data_h  = new double *[cpusize]; // CPU 接收暂存
    int length;

    // puts("Starting GPU transfer...");
    for (node = 0; node < cpusize; node++) {
        send_data_d[node] = rec_data_d[node] = nullptr;
        send_data_h[node] = rec_data_h[node] = nullptr;

        if (node == myrank) { // 同节点 D2D 直接拷贝，不涉及 MPI
            if ((length = gpu_data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))) {
                CUDA_CHECK(cudaMalloc((void**)&rec_data_d[node], length * sizeof(double)));
                gpu_data_packer(rec_data_d[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
            }
        }
        else { // 跨节点通信，走 CPU Staging
            // PACK 阶段
            if ((length = gpu_data_packer(0, src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry))) {
                CUDA_CHECK(cudaMalloc((void**)&send_data_d[node], length * sizeof(double)));
                send_data_h[node] = new double[length]; // 分配 CPU 内存

                // 1. GPU 内部完成打包
                gpu_data_packer(send_data_d[node], src[myrank], dst[myrank], node, PACK, VarList1, VarList2, Symmetry);
                
                // 2. D2H: 将打包好的连续数组拷回 CPU
                CUDA_CHECK(cudaMemcpy(send_data_h[node], send_data_d[node], length * sizeof(double), cudaMemcpyDeviceToHost));
                
                // 3. 将 CPU 指针交给 MPI (绝对安全)
                MPI_Isend((void *)send_data_h[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
            // UNPACK 的准备阶段
            if ((length = gpu_data_packer(0, src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry))) {
                CUDA_CHECK(cudaMalloc((void**)&rec_data_d[node], length * sizeof(double)));
                rec_data_h[node] = new double[length]; // 分配 CPU 内存
                
                // 接收到 CPU 内存
                MPI_Irecv((void *)rec_data_h[node], length, MPI_DOUBLE, node, 1, MPI_COMM_WORLD, reqs + req_no++);
            }
        }
    }

    // 等待所有 MPI 请求完成
    MPI_Waitall(req_no, reqs, stats);
    // puts("MPI communication completed, starting unpacking on GPU...");

    for (node = 0; node < cpusize; node++) {
        if (node == myrank) {
            if (rec_data_d[node]) {
                gpu_data_packer(rec_data_d[node], src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);
            }
        } 
        else {
            if (rec_data_h[node]) {
                // 获取需要拷贝的长度
                length = gpu_data_packer(0, src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);
                
                // 1. H2D: 将收到的 CPU 数据推上 GPU
                CUDA_CHECK(cudaMemcpy(rec_data_d[node], rec_data_h[node], length * sizeof(double), cudaMemcpyHostToDevice));
                
                // 2. GPU 内部完成解包映射
                gpu_data_packer(rec_data_d[node], src[node], dst[node], node, UNPACK, VarList1, VarList2, Symmetry);
            }
        }
    }
        
    // 释放所有资源
    for (node = 0; node < cpusize; node++) {
        if (send_data_d[node]) CUDA_CHECK(cudaFree(send_data_d[node]));
        if (rec_data_d[node]) CUDA_CHECK(cudaFree(rec_data_d[node]));
        if (send_data_h[node]) delete[] send_data_h[node];
        if (rec_data_h[node]) delete[] rec_data_h[node];
    }

    delete[] reqs;
    delete[] stats;
    delete[] send_data_d;
    delete[] rec_data_d;
    delete[] send_data_h;
    delete[] rec_data_h;
}
#endif

void Parallel::Sync_GPU(Patch *Pat, MyList<var> *VarList, int Symmetry) {
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_ghost_gsl(Pat); // ghost region only
    for (int node = 0; node < cpusize; node++) {
        src[node] = build_owned_gsl0(Pat, node);                              // for the part without ghost points and do not extend
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer_src[node], data locate on cpu#node;                                                                                                                                            // but for transfer_dst[node] the data may locate on any node
    }

    gpu_transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    if (dst) dst->destroyList();
    for (int node = 0; node < cpusize; node++) {
        if (src[node]) src[node]->destroyList();
        if (transfer_src[node]) transfer_src[node]->destroyList();
        if (transfer_dst[node]) transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}

void Parallel::Sync_GPU(MyList<Patch> *PatL, MyList<var> *VarList, int Symmetry) {
    // Patch inner Synch
    MyList<Patch> *Pp = PatL;
    while (Pp) {
        Sync_GPU(Pp->data, VarList, Symmetry);
        Pp = Pp->next;
    }

    // Patch inter Synch
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(PatL); // buffer region only
    for (int node = 0; node < cpusize; node++) {
        src[node] = build_owned_gsl(PatL, node, 5, Symmetry);                 // for the part without ghost nor buffer points and do not extend
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    gpu_transfer(transfer_src, transfer_dst, VarList, VarList, Symmetry);

    if (dst) dst->destroyList();
    for (int node = 0; node < cpusize; node++) {
        if (src[node]) src[node]->destroyList();
        if (transfer_src[node]) transfer_src[node]->destroyList();
        if (transfer_dst[node]) transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}

void Parallel::Restrict_GPU(
    MyList<Patch> *PatcL, MyList<Patch> *PatfL,
    MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
    int Symmetry
) {
    if (PatcL->data->lev >= PatfL->data->lev)
    {
        cout << "Parallel::Restrict: meet requst of Restrict from lev#" << PatfL->data->lev << " to lev#" << PatcL->data->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_complete_gsl(PatcL); // including ghost
    for (int node = 0; node < cpusize; node++)
    {
        // it seems bam always use this
        src[node] = build_owned_gsl(PatfL, node, 2, Symmetry); // - buffer - ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    gpu_transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}

void Parallel::OutBdLow2Hi_GPU(
    Patch *Patc, Patch *Patf,
    MyList<var> *VarList1 /* source */, MyList<var> *VarList2 /* target */,
    int Symmetry
) {
    if (Patc->lev >= Patf->lev)
    {
        cout << "Parallel::OutBdLow2Hi: meet requst of Prolong from lev#" << Patc->lev << " to lev#" << Patf->lev << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int cpusize;
    MPI_Comm_size(MPI_COMM_WORLD, &cpusize);

    MyList<Parallel::gridseg> *dst;
    MyList<Parallel::gridseg> **src, **transfer_src, **transfer_dst;
    src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_src = new MyList<Parallel::gridseg> *[cpusize];
    transfer_dst = new MyList<Parallel::gridseg> *[cpusize];

    dst = build_buffer_gsl(Patf); // buffer region only

    for (int node = 0; node < cpusize; node++)
    {
        src[node] = build_owned_gsl4(Patc, node, Symmetry);                   // - buffer - ghost - BD ghost
        build_gstl(src[node], dst, &transfer_src[node], &transfer_dst[node]); // for transfer[node], data locate on cpu#node
    }

    gpu_transfer(transfer_src, transfer_dst, VarList1, VarList2, Symmetry);

    if (dst)
        dst->destroyList();
    for (int node = 0; node < cpusize; node++)
    {
        if (src[node])
            src[node]->destroyList();
        if (transfer_src[node])
            transfer_src[node]->destroyList();
        if (transfer_dst[node])
            transfer_dst[node]->destroyList();
    }

    delete[] src;
    delete[] transfer_src;
    delete[] transfer_dst;
}

void Parallel::gpu_prepare_inter_time_level(
    Patch *Pat,
    MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
    MyList<var> *VarList3 /* target (t+a*dt) */, int tindex
) {
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MyList<var> *varl1;
    MyList<var> *varl2;
    MyList<var> *varl3;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            varl1 = VarList1;
            varl2 = VarList2;
            varl3 = VarList3;
            while (varl1)
            {
                if (tindex == 0) gpu_average_launch(cg->stream, cg->shape, cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varl2->data->sgfn], cg->d_fgfs[varl3->data->sgfn]);
                else if (tindex == 1) gpu_average3_launch(cg->stream, cg->shape, cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varl2->data->sgfn], cg->d_fgfs[varl3->data->sgfn]);
                else if (tindex == -1) gpu_average3_launch(cg->stream, cg->shape, cg->d_fgfs[varl2->data->sgfn], cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varl3->data->sgfn]);
                else {
                    cout << "error tindex in Parallel::prepare_inter_time_level" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                varl1 = varl1->next;
                varl2 = varl2->next;
                varl3 = varl3->next;
            }
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }
    GPUManager::getInstance().synchronize_all();
}

void Parallel::gpu_prepare_inter_time_level(
    Patch *Pat,
    MyList<var> *VarList1 /* source (t+dt) */, MyList<var> *VarList2 /* source (t) */,
    MyList<var> *VarList3 /* source (t-dt) */, MyList<var> *VarList4 /* target (t+a*dt) */, int tindex
) {
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MyList<var> *varl1;
    MyList<var> *varl2;
    MyList<var> *varl3;
    MyList<var> *varl4;

    MyList<Block> *BP = Pat->blb;
    while (BP)
    {
        Block *cg = BP->data;
        if (myrank == cg->rank)
        {
            varl1 = VarList1;
            varl2 = VarList2;
            varl3 = VarList3;
            varl4 = VarList4;
            while (varl1)
            {
                if (tindex == 0) gpu_average2_launch(
                    cg->stream, cg->shape, cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varl2->data->sgfn],
                    cg->d_fgfs[varl3->data->sgfn], cg->d_fgfs[varl4->data->sgfn]
                );
                else if (tindex == 1) gpu_average2p_launch(
                    cg->stream, cg->shape, cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varl2->data->sgfn],
                    cg->d_fgfs[varl3->data->sgfn], cg->d_fgfs[varl4->data->sgfn]
                );
                else if (tindex == -1) gpu_average2m_launch(
                    cg->stream, cg->shape, cg->fgfs[varl1->data->sgfn], cg->fgfs[varl2->data->sgfn],
                    cg->fgfs[varl3->data->sgfn], cg->fgfs[varl4->data->sgfn]
                );
                else {
                    cout << "error tindex in long cgh::prepare_inter_time_level" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                varl1 = varl1->next;
                varl2 = varl2->next;
                varl3 = varl3->next;
                varl4 = varl4->next;
            }
        }
        if (BP == Pat->ble)
            break;
        BP = BP->next;
    }
    GPUManager::getInstance().synchronize_all();
}