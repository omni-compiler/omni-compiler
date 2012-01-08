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
    OMP_PARALLEL_SECTIONS = 13 	/* parallel <clause_list> */
};

enum OMP_pragma_clause {
    OMP_DATA_DEFAULT = 0,
    OMP_DATA_PRIVATE = 1,		/* private <namelist> */
    OMP_DATA_SHARED = 2,		/* shared <namelist> */
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

    OMP_DIR_ORDERED=20,
    OMP_DIR_IF=21,
    OMP_DIR_NOWAIT=22,
    OMP_DIR_SCHEDULE=23
};

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
    OMP_DEFAULT_SHARED = 0
};

/* protype */
CExpr* lexParsePragmaOMP(char *p, int *token);
char *ompDirectiveName(int c);
char *ompClauseName(int c);

#endif
