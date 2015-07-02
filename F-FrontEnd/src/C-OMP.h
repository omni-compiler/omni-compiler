/*
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#ifndef _C_OMP_H
#define _C_OMP_H

enum OMP_pragma {
    OMP_PARALLEL = 0, 		/* parallel <clause_list> */
    OMP_FOR = 1,		/* loop <clause_list> */
    OMP_SECTIONS = 2,		/* sections <clause_list> */
    OMP_SECTION = 3,		/* section */
    OMP_SINGLE = 4,		/* single <clause list> */
    OMP_MASTER = 5,		/* master */
    OMP_CRITICAL = 6,		/* critical <name> */
    OMP_BARRIER = 7,		/* barrier */
    OMP_ATOMIC = 8,		/* atomic */
    OMP_FLUSH = 9,		/* flush <namelist> */
    OMP_ORDERED = 10,		/* ordered */
    OMP_THREADPRIVATE = 11,	/* threadprivate <namelist> */

    OMP_PARALLEL_FOR = 12, 	/* parallel <clause_list> */
    OMP_PARALLEL_SECTIONS = 13, 	/* parallel <clause_list> */
    
    OMP_TASK = 14,
    /*
     * Fortran entry
     */
    OMP_F_PARALLEL = 100,
    OMP_F_END_PARALLEL = 101,
    OMP_F_DO = 102,    
    OMP_F_END_DO = 103,	
    OMP_F_SECTIONS = 104,	
    OMP_F_END_SECTIONS = 105,	
    OMP_F_SECTION = 106,	
    OMP_F_SINGLE = 107,	
    OMP_F_END_SINGLE = 108,	
    OMP_F_MASTER = 110,	
    OMP_F_END_MASTER = 111,	
    OMP_F_CRITICAL = 112,	
    OMP_F_END_CRITICAL = 113,	
    OMP_F_BARRIER = 114,	
    OMP_F_ATOMIC = 115,	
    OMP_F_FLUSH = 116,	
    OMP_F_ORDERED = 117,		
    OMP_F_END_ORDERED = 118,		
    OMP_F_THREADPRIVATE = 119,	
    OMP_F_PARALLEL_DO = 120, 	
    OMP_F_END_PARALLEL_DO = 121, 	
    OMP_F_PARALLEL_SECTIONS = 122, 	
    OMP_F_END_PARALLEL_SECTIONS = 123,

    OMP_F_WORKSHARE = 130,
    OMP_F_END_WORKSHARE = 131,
    OMP_F_PARALLEL_WORKSHARE = 132,
    OMP_F_END_PARALLEL_WORKSHARE = 133,
    OMP_F_TASK = 134,
    OMP_F_END_TASK = 135,
};

enum OMP_pragma_clause {
    OMP_DATA_DEFAULT = 0,
    OMP_DATA_PRIVATE = 1,	/* private <namelist> */
    OMP_DATA_SHARED = 2,	/* shared <namelist> */
    OMP_DATA_FIRSTPRIVATE=3,
    OMP_DATA_LASTPRIVATE=4,
    OMP_DATA_COPYIN =5,

    OMP_DATA_REDUCTION_PLUS=6,
    OMP_DATA_REDUCTION_MINUS=7,
    OMP_DATA_REDUCTION_MUL=8,
    OMP_DATA_REDUCTION_BITAND=9,
    OMP_DATA_REDUCTION_BITOR=10,
    OMP_DATA_REDUCTION_BITXOR=11,
    OMP_DATA_REDUCTION_LOGAND=12,
    OMP_DATA_REDUCTION_LOGOR=13,
    OMP_DATA_REDUCTION_MIN=14,
    OMP_DATA_REDUCTION_MAX=15,
    OMP_DATA_REDUCTION_EQV=16,
    OMP_DATA_REDUCTION_NEQV=17,

    OMP_DATA_COPYPRIVATE=18,

    OMP_DIR_ORDERED=20,
    OMP_DIR_IF=21,
    OMP_DIR_NOWAIT=22,
    OMP_DIR_SCHEDULE=23,
    OMP_DIR_NUM_THREADS=24,
    OMP_DIR_TASK = 25,
    OMP_DIR_DEPEND = 26,

    OMP_DATA_DEPEND_IN =27,
    OMP_DATA_DEPEND_OUT =28,
    OMP_DATA_DEPEND_INOUT =29,

    OMP_DIR_UNTIED = 30,
    OMP_DIR_FINAL = 31,
    OMP_DIR_MERGEABLE = 32

};

#define IS_OMP_DATA_CLAUSE(c) \
(((int)OMP_DATA_PRIVATE) <= ((int)c) && \
 ((int)c) <= ((int)OMP_DATA_REDUCTION_NEQV))

#define IS_OMP_REDUCTION_DATA_CLAUSE(c) \
(((int)OMP_DATA_REDUCTION_PLUS) <= ((int)c) && \
 ((int)c) <= ((int)OMP_DATA_REDUCTION_NEQV))

enum OMP_sched_clause {
    OMP_SCHED_NONE = 0,
    OMP_SCHED_STATIC = 1,
    OMP_SCHED_DYNAMIC = 2,
    OMP_SCHED_GUIDED = 3,
    OMP_SCHED_RUNTIME = 4,
    OMP_SCHED_AFFINITY = 5
};

enum OMP_data_default {
    OMP_DEFAULT_NONE = 1,
    OMP_DEFAULT_SHARED = 0,
    OMP_DEFAULT_PRIVATE = 2
};
#endif
