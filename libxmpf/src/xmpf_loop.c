#include "xmpf_internal.h"


void xmpf_loop_sched__(int *lb, int *ub, int *st, int *r_idx, _XMP_object_ref_t **r_desc,
		       int *expand_type, int *lwidth, int *uwidth, int *unbound_flag)
{
  _XMP_object_ref_t *rp = *r_desc;
  _XMP_ASSERT(rp->ref_kind == XMP_OBJ_REF_TEMPL);
  _XMP_ASSERT(*expand_type != _XMP_LOOP_MARGIN || *lwidth == 0 || *uwidth == 0);
  
  int glb_orig = *lb;
  int gub_orig = *ub;

  if (rp->REF_INDEX[*r_idx] != -1){

    _XMP_ASSERT(*st != 0);

    _XMP_template_t *t_desc = rp->t_desc;
    int t_idx = rp->REF_INDEX[*r_idx];
    int off = rp->REF_OFFSET[*r_idx];  

    int global_ub_C = (*st > 0) ? (*ub + 1) : (*ub - 1);

    switch (t_desc->chunk[t_idx].dist_manner){

    case _XMP_N_DIST_DUPLICATION:
      _XMP_sched_loop_template_DUPLICATION(*lb + off, global_ub_C + off, *st,
					   lb, ub, st,
					   t_desc, t_idx);
      break;

    case _XMP_N_DIST_BLOCK:
      _XMP_sched_loop_template_BLOCK(*lb + off, global_ub_C + off, *st,
				     lb, ub, st,
				     t_desc, t_idx);
      break;

    case _XMP_N_DIST_CYCLIC:
      _XMP_sched_loop_template_CYCLIC(*lb + off, global_ub_C + off, *st,
				      lb, ub, st,
				      t_desc, t_idx);
      break;

    case _XMP_N_DIST_BLOCK_CYCLIC:
      _XMP_sched_loop_template_BLOCK_CYCLIC(*lb + off, global_ub_C + off, *st,
					    lb, ub, st,
					    t_desc, t_idx);
      break;

    case _XMP_N_DIST_GBLOCK:
      _XMP_sched_loop_template_GBLOCK(*lb + off, global_ub_C + off, *st,
				      lb, ub, st,
				      t_desc, t_idx);
      break;

    default:
      _XMP_fatal("xmpf_sched_loop_template: unknown chunk dist_manner");

    }

    (*ub)--; // because upper bound in Fortran is inclusive
  }
  else {
    ; /* the nest is not aligned with any dimension of the template. */
  }

  if (*expand_type == _XMP_LOOP_NONE){
    //xmpf_dbg_printf("%d : %d\n", *lb, *ub);
    return;
  }
  else if (*expand_type == _XMP_LOOP_EXPAND){

    if ((*lb) <= (*ub)){ // iterates at least once
      (*lb) -= (*lwidth);
      (*ub) += (*uwidth);
    }

  }
  else if (*expand_type == _XMP_LOOP_MARGIN){

    if ((*lb) <= (*ub)){ // iterates at least once

      if (*lwidth > 0){
	(*lb) -= (*lwidth);
	(*ub) = (*lb) + (*lwidth) - 1;
      }
      else if (*lwidth < 0){
	(*ub) = (*lb) - (*lwidth) - 1;
	// (*lb)
      }
      else if (*uwidth > 0){
	(*ub) += (*uwidth);
	(*lb) = (*ub) - (*uwidth) + 1;
      }
      else if (*uwidth < 0){
	(*lb) = (*ub) + (*uwidth) + 1;
	// (*ub)
      }

    }

  }

  if (*unbound_flag > 0){

    _XMP_template_t *t_desc = rp->t_desc;
    int t_idx = rp->REF_INDEX[*r_idx];

    long long int glb;
    _XMP_L2G(*lb, &glb, t_desc, t_idx);
    if (glb < glb_orig){
      (*lb) += (*lwidth);
    }

    long long int gub;
    _XMP_L2G(*ub, &gub, t_desc, t_idx);
    if (gub > gub_orig){
      (*ub) -= (*uwidth);
    }
  }

  //xmpf_dbg_printf("%d : %d\n", *lb, *ub);
  
  return;

}


int xmpf_loop_test_skip__(_XMP_object_ref_t **r_desc, int *rdim, int *i)
{
  _XMP_object_ref_t *rp = *r_desc;
/*   int index = *i + rp->offset[*rdim]; */
/*   int tdim = rp->offset[*rdim]; */
  int index = *i + rp->REF_OFFSET[*rdim];
  int tdim = rp->REF_INDEX[*rdim];

  _XMP_template_t *tp = rp->t_desc;
  _XMP_template_chunk_t *cp = &tp->chunk[tdim];
  long long base = tp->info[tdim].ser_lower;
  _XMP_nodes_info_t *n_info = cp->onto_nodes_info;

  switch(cp->dist_manner){
  case _XMP_N_DIST_DUPLICATION:
    return 0;
  case _XMP_N_DIST_BLOCK:
  case _XMP_N_DIST_GBLOCK:
    // par_lower and par_upper is 0-origin. How about index ?
    return (index < cp->par_lower || cp->par_upper < index);
  case _XMP_N_DIST_CYCLIC:
    //xmpf_dbg_printf("%d, %d, %d, %d\n", index, base, n_info->size, n_info->rank);
    return ((index - base) % n_info->size != n_info->rank);
  case _XMP_N_DIST_BLOCK_CYCLIC:
    {
      int w = cp->par_width;
      //xmpf_dbg_printf("%d, %d, %d, %d\n", index, base, n_info->size, w);
      return ((index - base) % (w * n_info->size) / w != n_info->rank);
    }
  default:
    _XMP_fatal("_xmpf_loop_test_skip_: unknown chunk dist_manner");
    return 0; /* dummy */
  }
}


/* int xmpf_loop_test_1__(_XMP_object_ref_t **r_desc, int *i) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */
/*     int index = *i - rp->offset[0]; */
/*     _XMP_template_t *tp = rp->t_desc; */
/*     _XMP_template_chunk_t *cp = &tp->chunk[0]; */
/*     return (cp->par_lower <= index && index <= cp->par_upper); */
/* } */


/* int xmpf_loop_test_2_(_XMP_object_ref_t **r_desc, int *i1, int *i0) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */
/*     int index0 = *i0 - rp->offset[0]; */
/*     int index1 = *i1 - rp->offset[1]; */
/*     _XMP_template_t *tp = rp->t_desc; */
/*     _XMP_template_chunk_t *cp0 = &tp->chunk[0]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[1]; */
/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper); */
/* } */

/*
int xmpf_loop_test_2__(_XMP_object_ref_t **r_desc, int *i0, int *i1)
{
    _XMP_object_ref_t *rp = *r_desc;

    int index0 = *i0 + rp->offset[0];
    int index1 = *i1 + rp->offset[1];

    _XMP_template_t *tp = rp->t_desc;

    _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]];
    _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]];

    return (cp0->par_lower <= index0 && index0 <= cp0->par_upper &&
	    cp1->par_lower <= index1 && index1 <= cp1->par_upper);
}
*/

/* int xmpf_loop_test_3__(_XMP_object_ref_t **r_desc, int *i0, int *i1, int *i2) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */

/*     int index0 = *i0 + rp->offset[0]; */
/*     int index1 = *i1 + rp->offset[1]; */
/*     int index2 = *i2 + rp->offset[2]; */

/*     _XMP_template_t *tp = rp->t_desc; */

/*     _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]]; */
/*     _XMP_template_chunk_t *cp2 = &tp->chunk[rp->index[2]]; */

/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper && */
/* 	    cp2->par_lower <= index2 && index2 <= cp2->par_upper); */
/* } */


/* int xmpf_loop_test_4__(_XMP_object_ref_t **r_desc, int *i0, int *i1, int *i2, int *i3) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */

/*     int index0 = *i0 + rp->offset[0]; */
/*     int index1 = *i1 + rp->offset[1]; */
/*     int index2 = *i2 + rp->offset[2]; */
/*     int index3 = *i3 + rp->offset[3]; */

/*     _XMP_template_t *tp = rp->t_desc; */

/*     _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]]; */
/*     _XMP_template_chunk_t *cp2 = &tp->chunk[rp->index[2]]; */
/*     _XMP_template_chunk_t *cp3 = &tp->chunk[rp->index[3]]; */

/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper && */
/* 	    cp2->par_lower <= index2 && index2 <= cp2->par_upper && */
/* 	    cp3->par_lower <= index3 && index3 <= cp3->par_upper); */
/* } */


/* int xmpf_loop_test_5__(_XMP_object_ref_t **r_desc, int *i0, int *i1, int *i2, int *i3, */
/* 		                                   int *i4) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */

/*     int index0 = *i0 + rp->offset[0]; */
/*     int index1 = *i1 + rp->offset[1]; */
/*     int index2 = *i2 + rp->offset[2]; */
/*     int index3 = *i3 + rp->offset[3]; */
/*     int index4 = *i4 + rp->offset[4]; */

/*     _XMP_template_t *tp = rp->t_desc; */

/*     _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]]; */
/*     _XMP_template_chunk_t *cp2 = &tp->chunk[rp->index[2]]; */
/*     _XMP_template_chunk_t *cp3 = &tp->chunk[rp->index[3]]; */
/*     _XMP_template_chunk_t *cp4 = &tp->chunk[rp->index[4]]; */

/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper && */
/* 	    cp2->par_lower <= index2 && index2 <= cp2->par_upper && */
/* 	    cp3->par_lower <= index3 && index3 <= cp3->par_upper && */
/* 	    cp4->par_lower <= index4 && index4 <= cp4->par_upper); */
/* } */


/* int xmpf_loop_test_6__(_XMP_object_ref_t **r_desc, int *i0, int *i1, int *i2, int *i3, */
/* 		                                   int *i4, int *i5) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */

/*     int index0 = *i0 + rp->offset[0]; */
/*     int index1 = *i1 + rp->offset[1]; */
/*     int index2 = *i2 + rp->offset[2]; */
/*     int index3 = *i3 + rp->offset[3]; */
/*     int index4 = *i4 + rp->offset[4]; */
/*     int index5 = *i5 + rp->offset[5]; */

/*     _XMP_template_t *tp = rp->t_desc; */

/*     _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]]; */
/*     _XMP_template_chunk_t *cp2 = &tp->chunk[rp->index[2]]; */
/*     _XMP_template_chunk_t *cp3 = &tp->chunk[rp->index[3]]; */
/*     _XMP_template_chunk_t *cp4 = &tp->chunk[rp->index[4]]; */
/*     _XMP_template_chunk_t *cp5 = &tp->chunk[rp->index[5]]; */

/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper && */
/* 	    cp2->par_lower <= index2 && index2 <= cp2->par_upper && */
/* 	    cp3->par_lower <= index3 && index3 <= cp3->par_upper && */
/* 	    cp4->par_lower <= index4 && index4 <= cp4->par_upper && */
/* 	    cp5->par_lower <= index5 && index5 <= cp5->par_upper); */
/* } */


/* int xmpf_loop_test_7__(_XMP_object_ref_t **r_desc, int *i0, int *i1, int *i2, int *i3, */
/* 		                                   int *i4, int *i5, int *i6) */
/* { */
/*     _XMP_object_ref_t *rp = *r_desc; */

/*     int index0 = *i0 + rp->offset[0]; */
/*     int index1 = *i1 + rp->offset[1]; */
/*     int index2 = *i2 + rp->offset[2]; */
/*     int index3 = *i3 + rp->offset[3]; */
/*     int index4 = *i4 + rp->offset[4]; */
/*     int index5 = *i5 + rp->offset[5]; */
/*     int index6 = *i6 + rp->offset[6]; */

/*     _XMP_template_t *tp = rp->t_desc; */

/*     _XMP_template_chunk_t *cp0 = &tp->chunk[rp->index[0]]; */
/*     _XMP_template_chunk_t *cp1 = &tp->chunk[rp->index[1]]; */
/*     _XMP_template_chunk_t *cp2 = &tp->chunk[rp->index[2]]; */
/*     _XMP_template_chunk_t *cp3 = &tp->chunk[rp->index[3]]; */
/*     _XMP_template_chunk_t *cp4 = &tp->chunk[rp->index[4]]; */
/*     _XMP_template_chunk_t *cp5 = &tp->chunk[rp->index[5]]; */
/*     _XMP_template_chunk_t *cp6 = &tp->chunk[rp->index[6]]; */

/*     return (cp0->par_lower <= index0 && index0 <= cp0->par_upper && */
/* 	    cp1->par_lower <= index1 && index1 <= cp1->par_upper && */
/* 	    cp2->par_lower <= index2 && index2 <= cp2->par_upper && */
/* 	    cp3->par_lower <= index3 && index3 <= cp3->par_upper && */
/* 	    cp4->par_lower <= index4 && index4 <= cp4->par_upper && */
/* 	    cp5->par_lower <= index5 && index5 <= cp5->par_upper && */
/* 	    cp6->par_lower <= index6 && index6 <= cp6->par_upper); */
/* } */
