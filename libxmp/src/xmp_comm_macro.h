/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

#ifndef _XCALABLEMP_COMM_MACRO
#define _XCALABLEMP_COMM_MACRO

// reduce
#define _XCALABLEMP_M_REDUCE_EXEC(addr, count, datatype, op) \
_XCALABLEMP_reduce_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op)

#define _XCALABLEMP_M_REDUCE_FLMM_EXEC(addr, count, datatype, op, num_locs, ...) \
_XCALABLEMP_reduce_FLMM_NODES_ENTIRE(_XCALABLEMP_get_execution_nodes(), addr, count, datatype, op, num_locs, __VA_ARGS__)

// bcast
#define _XCALABLEMP_M_BCAST_EXEC_OMITTED(addr, count, datatype_size) \
_XCALABLEMP_bcast_NODES_ENTIRE_OMITTED(_XCALABLEMP_get_execution_nodes(), addr, count, datatype_size)

#define _XCALABLEMP_M_BCAST_EXEC_GLOBAL(addr, count, datatype_size, from_l, from_u, from_s) \
_XCALABLEMP_bcast_NODES_ENTIRE_GLOBAL(_XCALABLEMP_get_execution_nodes(), addr, count, datatype_size, from_l, from_u, from_s)

#define _XCALABLEMP_M_BCAST_EXEC_NODES(addr, count, datatype_size, from_nodes, ...) \
_XCALABLEMP_bcast_NODES_ENTIRE_NODES(_XCALABLEMP_get_execution_nodes(), addr, count, datatype_size, from_nodes, __VA_ARGS__)

#endif // _XCALABLEMP_COMM_MACRO
