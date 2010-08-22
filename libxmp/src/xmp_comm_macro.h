#ifndef _XCALABLEMP_COMM_MACRO
#define _XCALABLEMP_COMM_MACRO

#define _XCALABLEMP_M_REDUCE_EXEC(addr, count, datatype, op) \
_XCALABLEMP_reduce_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op)

#define _XCALABLEMP_M_REDUCE_FLMM_EXEC(addr, count, datatype, op, num_locs, ...) \
_XCALABLEMP_reduce_FLMM_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op, num_locs, __VA_ARGS__)

#endif // _XCALABLEMP_COMM_MACRO
