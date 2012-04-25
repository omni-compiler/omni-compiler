#include "xmpf_internal.h"

void xmpf_sched_loop_template_local__(int *global_lb, int *global_ub, int *global_step,
				      int *local_lb, int *local_ub, int *local_step,
				      _XMP_template_t **t_desc, int *t_idx, int *off)
{

  _XMP_ASSERT(*global_step != 0);
  int global_ub_C = (*global_step > 0) ? (*global_ub + 1) : (*global_ub - 1);

  switch ((*t_desc)->chunk[*t_idx].dist_manner){

    case _XMP_N_DIST_DUPLICATION:
      _XMP_sched_loop_template_DUPLICATION(*global_lb, global_ub_C, *global_step,
					   local_lb, local_ub, local_step,
					   *t_desc, *t_idx);
      break;

    case _XMP_N_DIST_BLOCK:
      _XMP_sched_loop_template_BLOCK(*global_lb, global_ub_C, *global_step,
				     local_lb, local_ub, local_step,
				     *t_desc, *t_idx);
      break;

    case _XMP_N_DIST_CYCLIC:
      _XMP_sched_loop_template_CYCLIC(*global_lb, global_ub_C, *global_step,
				      local_lb, local_ub, local_step,
				      *t_desc, *t_idx);
      break;

    case _XMP_N_DIST_BLOCK_CYCLIC:
      _XMP_sched_loop_template_BLOCK_CYCLIC(*global_lb, global_ub_C, *global_step,
					    local_lb, local_ub, local_step,
					    *t_desc, *t_idx);
      break;

    default:
      _XMP_fatal("xmpf_sched_loop_template: unknown chunk dist_manner");

  }

  (*local_ub)--; // because upper bound in Fortran is inclusive

/*   xmpf_dbg_printf("loop_sched(%d,%d,%d)->(%d,%d,%d)\n", */
/* 		  *global_lb,*global_ub,*global_step, */
/* 		  *local_lb,*local_ub,*local_step); */

  return;

}


int xmpf_loop_test_1_(_XMP_object_ref_t **r_desc, int *i)
{
    _XMP_object_ref_t *rp = *r_desc;
    int index = *i - rp->offset[0];
    _XMP_template_t *tp = rp->t_desc;
    _XMP_template_chunk_t *cp = &tp->chunk[0];
    return (cp->par_lower <= index && index <= cp->par_upper);
}


int xmpf_loop_test_2_(_XMP_object_ref_t **r_desc, int *i1, int *i0)
{
    _XMP_object_ref_t *rp = *r_desc;
    int index0 = *i0 - rp->offset[0];
    int index1 = *i1 - rp->offset[1];
    _XMP_template_t *tp = rp->t_desc;
    _XMP_template_chunk_t *cp0 = &tp->chunk[0];
    _XMP_template_chunk_t *cp1 = &tp->chunk[1];
    return (cp0->par_lower <= index0 && index0 <= cp0->par_upper &&
	    cp1->par_lower <= index1 && index1 <= cp1->par_upper);
}
