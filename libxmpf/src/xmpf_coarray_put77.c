/*
 *   COARRAY
 *   Fortran77 interface put routtine
 *
 *   arguments:
 *     for rank r=0:
 *       xmpf_coarray_put77d0_
 *          ( descID,    elemLen,     baseAddr,          destNode, RHS )
 *     for rank r<=7:
 *       xmpf_coarray_put77dr_
 *          ( descID,    elemLen,     baseAddr,
 *            size(0),   stride(0),   neighborAddr(1),
 *            ...
 *            size(r-1), stride(r-1), neighborAddr(r-1), destNode, RHS )
 *
 *   important point:
 *     If neighborAddr(1) - baseAddr == elemLen,
 *       it is contiguous upto 1st dimension.
 *     If neighborAddr(2) - baseAddr == elemLen * size(0),
 *       it is contiguous upto 2nd dimension.
 *     ...
 */


#include "xmpf_internal.h"

static void _coarray_put77d0(void *co_desc, int costart, int length,
                             int elem, int node, void *localAddr);


// always contiguous
void xmpf_coarray_put77d0_(int *descrId, int *elemLen, void *baseAddr,
                           int *coindex, void *rhs )
{
  coarray_info_t *info = get_coarray_info(*descrId);
  int costart = (baseAddr - info->co_addr) / info->elem_size;
  int length = *elemLen / info->elem_size;

  _coarray_put77d0(info->co_desc, costart, length,
                   info->elem_size, *coindex - 1, rhs);
}

void _coarray_put77d0(void *co_desc, int costart, int length,
                      int elem, int node, void *localAddr)
{
  _XMP_coarray_rdma_coarray_set_1(costart, length, 1);
  _XMP_coarray_rdma_array_set_1(0, length, 1, length, elem);
  _XMP_coarray_rdma_node_set_1(node);
  _XMP_coarray_rdma_do(CO_RDMA_PUT, co_desc, localAddr, 0);
}

