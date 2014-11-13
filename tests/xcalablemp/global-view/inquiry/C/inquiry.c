#include <stdio.h>
#include <stdlib.h>
#include "xmp.h"

int a[6][9][16], a1[6], b[6];
int m[2]={2,4};
int gidx[3]={3,3,3}, lidx[3];
#pragma xmp nodes p(2,3,2)
#pragma xmp nodes p1(2)=p(1:2,1,1)
#pragma xmp template t(0:15,0:5,0:8)
#pragma xmp template t1(0:5)
#pragma xmp template t2(:)
#pragma xmp distribute t(block, cyclic, cyclic(2)) onto p
#pragma xmp distribute t1(gblock(m)) onto p1
#pragma xmp distribute t2(block) onto p1
#pragma xmp align a[i0][i1][i2] with t(i2,i0,i1)
#pragma xmp align a1[i0] with t1(i0)
#pragma xmp align b[i0] with t(i0,*,*)
#pragma xmp shadow a[0][0][1:2]

void check(int irslt, int ians, int *error){

  //printf("irslt=%d,ians=%d\n",irslt,ians);
  if(irslt != ians){
    *error = *error + 1;
  }

}

int main(){

  int irank, ierr, error=0;
  xmp_desc_t dt, dn, dt1, dn1, dn2;
  int ival,lb[3],ub[3],st[3],map[2];

  ierr=xmp_align_template(xmp_desc_of(a), &dt);
  ierr=xmp_dist_nodes(dt, &dn);
  ierr=xmp_align_template(xmp_desc_of(a1), &dt1);
  ierr=xmp_dist_nodes(dt1, &dn1);

  irank=xmp_node_num();

  ierr=xmp_template_fixed(xmp_desc_of(t2), &ival);
  check(ival, 0, &error);

#pragma xmp template_fix(block) t2(0:5)

  if (irank==11){
    ierr=xmp_nodes_ndims(dn, &ival);
    check(ival, 3, &error);

    ierr=xmp_nodes_index(dn, 1, &ival);
    check(ival, 1, &error);
    ierr=xmp_nodes_index(dn, 2, &ival);
    check(ival, 3, &error);
    ierr=xmp_nodes_index(dn, 3, &ival);
    check(ival, 2, &error);

    ierr=xmp_nodes_size(dn, 1, &ival);
    check(ival, 2, &error);
    ierr=xmp_nodes_size(dn, 2, &ival);
    check(ival, 3, &error);
    ierr=xmp_nodes_size(dn, 3, &ival);
    check(ival, 2, &error);

    ierr=xmp_nodes_equiv(dn1, &dn2, lb, ub, st);
    check(lb[0], 1, &error);
    check(ub[0], 2, &error);
    check(st[0], 1, &error);

    /*ierr=xmp_nodes_attr(dn, &ival);
    check(ival, 3, &error);
    ierr=xmp_nodes_attr(dn1, &ival);
    check(ival, 3, &error);*/

    ierr=xmp_template_fixed(xmp_desc_of(t2), &ival);
    check(ival, 1, &error);
    ierr=xmp_template_ndims(dt, &ival);
    check(ival, 3, &error);
    ierr=xmp_template_lbound(dt, 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_template_lbound(dt, 2, &ival);
    check(ival, 0, &error);
    ierr=xmp_template_lbound(dt, 3, &ival);
    check(ival, 0, &error);
    ierr=xmp_template_ubound(dt, 1, &ival);
    check(ival, 15, &error);
    ierr=xmp_template_ubound(dt, 2, &ival);
    check(ival, 5, &error);
    ierr=xmp_template_ubound(dt, 3, &ival);
    check(ival, 8, &error);
    ierr=xmp_dist_format(dt, 1, &ival);
    check(ival, 2101, &error);
    ierr=xmp_dist_format(dt, 2, &ival);
    check(ival, 2102, &error);
    ierr=xmp_dist_format(dt, 3, &ival);
    check(ival, 2102, &error);
    ierr=xmp_dist_blocksize(dt, 1, &ival);
    check(ival, 8, &error);
    ierr=xmp_dist_blocksize(dt, 2, &ival);
    check(ival, 1, &error);
    ierr=xmp_dist_blocksize(dt, 3, &ival);
    check(ival, 2, &error);

    ierr=xmp_dist_gblockmap(dt1, 1, map);
    check(map[0], 2, &error);
    check(map[1], 4, &error);
    ierr=xmp_dist_axis(dt, 1, &ival);
    check(ival, 1, &error);
    ierr=xmp_dist_axis(dt, 2, &ival);
    check(ival, 2, &error);
    ierr=xmp_dist_axis(dt, 3, &ival);
    check(ival, 3, &error);

    ierr=xmp_align_axis(xmp_desc_of(a), 1, &ival);
    check(ival, 2, &error);
    ierr=xmp_align_axis(xmp_desc_of(a), 2, &ival);
    check(ival, 3, &error);
    ierr=xmp_align_axis(xmp_desc_of(a), 3, &ival);
    check(ival, 1, &error);

    ierr=xmp_align_offset(xmp_desc_of(a), 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_align_offset(xmp_desc_of(a), 2, &ival);
    check(ival, 0, &error);
    ierr=xmp_align_offset(xmp_desc_of(a), 3, &ival);
    check(ival, 0, &error);

    ierr=xmp_align_replicated(xmp_desc_of(b), 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_align_replicated(xmp_desc_of(b), 2, &ival);
    check(ival, 1, &error);
    ierr=xmp_align_replicated(xmp_desc_of(b), 3, &ival);
    check(ival, 1, &error);

    ierr=xmp_array_ndims(xmp_desc_of(a), &ival);
    check(ival, 3, &error);
    ierr=xmp_array_lbound(xmp_desc_of(a), 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_lbound(xmp_desc_of(a), 2, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_lbound(xmp_desc_of(a), 3, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_ubound(xmp_desc_of(a), 1, &ival);
    check(ival, 5, &error);
    ierr=xmp_array_ubound(xmp_desc_of(a), 2, &ival);
    check(ival, 8, &error);
    ierr=xmp_array_ubound(xmp_desc_of(a), 3, &ival);
    check(ival, 15, &error);
    ierr=xmp_array_lshadow(xmp_desc_of(a), 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_lshadow(xmp_desc_of(a), 2, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_lshadow(xmp_desc_of(a), 3, &ival);
    check(ival, 1, &error);
    ierr=xmp_array_ushadow(xmp_desc_of(a), 1, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_ushadow(xmp_desc_of(a), 2, &ival);
    check(ival, 0, &error);
    ierr=xmp_array_ushadow(xmp_desc_of(a), 3, &ival);
    check(ival, 2, &error);
    ierr=xmp_array_gtol(xmp_desc_of(a), gidx, lidx);
    check(lidx[0], 1, &error);
    check(lidx[1], 1, &error);
    check(lidx[2], 4, &error);


    if(error == 0){
      printf("PASS\n");
    }
    else{
      fprintf(stderr, "ERROR count=%d\n",error);
      exit(1);
    }

    return 0;

  }

}

