#include "helper.h"

#include "MyList.h"
#include "MPatch.h"
#include "Block.h"
#include "var.h"
namespace Helper {

void move_to_gpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList) {
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
				cg->move_to_gpu(VarList);
            }
            BP = BP->next;
        }
        Pp = Pp->next;
    }
}

void move_to_cpu_whole(MyList<Patch> *Pp, int myrank, MyList<var> *VarList) {
    while (Pp) {
        MyList<Block> *BP = Pp->data->blb;
        while (BP) {
            Block *cg = BP->data;
            if (myrank == cg->rank) {
				cg->move_to_cpu(VarList);
            }
            BP = BP->next;
        }
        Pp = Pp->next;
    }
}

}