#ifndef _XMP_COMM_MACRO
#define _XMP_COMM_MACRO

// reduce
#define _XMP_M_REDUCE_EXEC(addr, count, datatype, op) \
_XMP_reduce_NODES_ENTIRE(_XMP_get_execution_nodes(), addr, count, datatype, op)

#define _XMP_M_REDUCE_FLMM_EXEC(addr, count, datatype, op, num_locs, ...) \
_XMP_reduce_FLMM_NODES_ENTIRE(_XMP_get_execution_nodes(), addr, count, datatype, op, num_locs, __VA_ARGS__)

// reduce acc
#define _XMP_M_REDUCE_ACC_EXEC(addr, count, datatype, op) \
_XMP_reduce_acc_NODES_ENTIRE(_XMP_get_execution_nodes(), addr, count, datatype, op)

#define _XMP_M_REDUCE_ACC_FLMM_EXEC(addr, count, datatype, op, num_locs, ...) \
_XMP_reduce_acc_FLMM_NODES_ENTIRE(_XMP_get_execution_nodes(), addr, count, datatype, op, num_locs, __VA_ARGS__)

// bcast
#define _XMP_M_BCAST_EXEC_OMITTED(addr, count, datatype_size) \
_XMP_bcast_NODES_ENTIRE_OMITTED(_XMP_get_execution_nodes(), addr, count, datatype_size)

#define _XMP_M_BCAST_EXEC_GLOBAL(addr, count, datatype_size, from_l, from_u, from_s) \
_XMP_bcast_NODES_ENTIRE_GLOBAL(_XMP_get_execution_nodes(), addr, count, datatype_size, from_l, from_u, from_s)

#define _XMP_M_BCAST_EXEC_NODES(addr, count, datatype_size, from_nodes, ...) \
_XMP_bcast_NODES_ENTIRE_NODES(_XMP_get_execution_nodes(), addr, count, datatype_size, from_nodes, __VA_ARGS__)

// bcast acc
#define _XMP_M_BCAST_ACC_EXEC_OMITTED(addr, count, datatype_size) \
_XMP_bcast_acc_NODES_ENTIRE_OMITTED(_XMP_get_execution_nodes(), addr, count, datatype_size)

#define _XMP_M_BCAST_ACC_EXEC_GLOBAL(addr, count, datatype_size, from_l, from_u, from_s) \
_XMP_bcast_acc_NODES_ENTIRE_GLOBAL(_XMP_get_execution_nodes(), addr, count, datatype_size, from_l, from_u, from_s)

#define _XMP_M_BCAST_ACC_EXEC_NODES(addr, count, datatype_size, from_nodes, ...) \
_XMP_bcast_acc_NODES_ENTIRE_NODES(_XMP_get_execution_nodes(), addr, count, datatype_size, from_nodes, __VA_ARGS__)

#endif // _XMP_COMM_MACRO
