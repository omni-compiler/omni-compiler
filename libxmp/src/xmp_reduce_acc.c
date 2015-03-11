#include <xmp_internal.h>
extern void _XMP_reduce_gpu_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op);
extern void _XMP_reduce_gpu_CLAUSE(void *data_addr, int count, int datatype, int op);
void _XMP_reduce_acc_NODES_ENTIRE(_XMP_nodes_t *nodes, void *data_addr, int count, int datatype, int op)
{
  _XMP_reduce_gpu_NODES_ENTIRE(nodes, data_addr, count, datatype, op);
}

void _XMP_reduce_acc_FLMM_NODES_ENTIRE(_XMP_nodes_t *nodes, void *addr, int count, int datatype, int op, int num_locs, ...)
{

}

void _XMP_reduce_acc_CLAUSE(void *data_addr, int count, int datatype, int op)
{
  _XMP_reduce_gpu_CLAUSE(data_addr, count, datatype, op);
}

void _XMP_reduce_acc_FLMM_CLAUSE(void *data_addr, int count, int datatype, int op, int num_locs, ...)
{

}
