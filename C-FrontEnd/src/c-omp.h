#ifndef _C_OMP_H
#define _C_OMP_H

enum OMP_pragma {
    OMP_PARALLEL = 0, 		/* parallel <clause_list> */
    OMP_FOR = 1,		/* loop <clause_list> */
    OMP_SECTIONS = 2,		/* sections <clause_list> */
    OMP_SECTION = 3,		/* section */
    OMP_SINGLE = 4,		/* single <clause list> */
    OMP_MASTER = 5,		/* master */
    OMP_CRITICAL = 6,		/* critical <name> <hint> */
    OMP_BARRIER = 7,		/* barrier */
    OMP_ATOMIC = 8,		/* atomic */
    OMP_FLUSH = 9,		/* flush <namelist> */
    OMP_ORDERED = 10,		/* ordered <clause> */
    OMP_THREADPRIVATE = 11,	/* threadprivate <namelist> */

    OMP_PARALLEL_FOR = 12, 	/* parallel <clause_list> */
    OMP_PARALLEL_SECTIONS = 13,	/* parallel <clause_list> */

    OMP_SIMD = 14,		/* simd <clause_list> */
    OMP_DECLARE_SIMD = 15,	/* declare simd <clause_list> */
    OMP_LOOP_SIMD = 16,		/* for simd <clause_list> */
    OMP_TASK = 17,		/* task <clause_list> */
    OMP_TASKLOOP = 18,		/* taskloop <clause_list> */
    OMP_TASKLOOP_SIMD = 19,	/* taskloop simd <clause_list> */
    OMP_TASKYIELD = 20,		/* taskyield */
    OMP_TARGET_DATA = 21,	/* target data <clause_list> */
    OMP_TARGET_ENTER_DATA = 22,	/* target enter data <clause_list> */
    OMP_TARGET_EXIT_DATA = 23,	/* target exit data <clause_list> */
    OMP_TARGET = 24,		/* target <clause_list> */
    OMP_TARGET_UPDATE = 25,	/* target update <clause_list> */
    OMP_DECLARE_TARGET = 26,	/* declare target <extended-list> <clause_list> */
    OMP_TEAMS = 27,		/* teams <clause_list> */
    OMP_DISTRIBUTE = 28,	/* distribute <clause_list> */
    OMP_DISTRIBUTE_SIMD = 29,	/* distribute simd <clause_list> */
    OMP_DISTRIBUTE_PARALLEL_LOOP = 30,			/* distribute parallel for <clause_list> */
    OMP_DISTRIBUTE_PARALLEL_LOOP_SIMD = 31,		/* distribute parallel for simd <clause_list> */
    OMP_PARALLEL_LOOP_SIMD = 32,			/* parallel for simd <clause_list> */
    OMP_TARGET_PARALLEL = 33,				/* target parallel <clause_list> */
    OMP_TARGET_PARALLEL_LOOP = 34,			/* target parallel for <clause_list> */
    OMP_TARGET_PARALLEL_LOOP_SIMD = 35,			/* target parallel for simd <clause_list> */
    OMP_TARGET_SIMD = 36,				/* target simd <clause_list> */
    OMP_TARGET_TEAMS = 37,				/* target teams <clause_list> */
    OMP_TEAMS_DISTRIBUTE = 38,				/* teams distribute <clause_list> */
    OMP_TEAMS_DISTRIBUTE_SIMD = 39,			/* teams distribute simd <clause_list> */
    OMP_TARGET_TEAMS_DISTRIBUTE = 40,			/* target teams distribute <clause_list> */
    OMP_TARGET_TEAMS_DISTRIBUTE_SIMD = 41,		/* target teams distribute simd <clause_list> */
    OMP_TEAMS_DISTRIBUTE_PARALLEL_LOOP = 42,		/* teams distribute parallel for <clause_list> */
    OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP = 43,	/* target teams distribute parallel for <clause_list> */
    OMP_TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD = 44,	/* teams distribute parallel for simd <clause_list> */
    OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP_SIMD = 45,/* target teams distribute parallel for simd <clause_list> */

    OMP_TASKWAIT = 46,		/* taskwait */
    OMP_TASKGROUP = 47,		/* taskgroup */
    OMP_CANCEL= 48,		/* cancel <type> <if-clause> */
    OMP_CANCELLATION_POINT= 49,	/* cancellation point <type> */
    OMP_DECLARE_REDUCTION= 50,	/* declare reduction <id> <type-list> <combiner> <init-clause> */
};

#define IS_OMP_PRAGMA_CODE(code) (((int)(code)) < 100)

enum OMP_pragma_clause {
    OMP_CLAUSE_NONE = 0, // reserved for none
    OMP_DATA_DEFAULT = 1,
    OMP_DATA_PRIVATE = 2,		/* private <namelist> */
    OMP_DATA_SHARED = 3,		/* shared <namelist> */
    OMP_DATA_FIRSTPRIVATE=4,
    OMP_DATA_LASTPRIVATE=5,
    OMP_DATA_COPYIN =6,

    OMP_DATA_REDUCTION_PLUS	=10,
    OMP_DATA_REDUCTION_MINUS	=11,
    OMP_DATA_REDUCTION_MUL	=12,
    OMP_DATA_REDUCTION_BITAND	=13,
    OMP_DATA_REDUCTION_BITOR	=14,
    OMP_DATA_REDUCTION_BITXOR	=15,
    OMP_DATA_REDUCTION_LOGAND	=16,
    OMP_DATA_REDUCTION_LOGOR	=17,
    OMP_DATA_REDUCTION_MIN	=18,
    OMP_DATA_REDUCTION_MAX	=19,

    OMP_DIR_ORDERED=20,
    OMP_DIR_IF=21,
    OMP_DIR_NOWAIT=22,
    OMP_DIR_SCHEDULE=23,
    OMP_DIR_NUM_THREADS=24,
    
    OMP_COLLAPSE=25,
    OMP_DEPEND=26,

    OMP_DATA_LINEAR=27,
    OMP_DATA_COPYPRIVATE=28,
    OMP_DATA_MAP=29,
    OMP_DATA_DEFAULT_MAP=30,
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
    OMP_DEFAULT_NONE = 0,
    OMP_DEFAULT_SHARED = 1,
    OMP_DEFAULT_PRIVATE = 2
};

/* protype */
char *ompDirectiveName(int c);
char *ompClauseName(int c);
char *ompScheduleName(int c);
char *ompDataDefaultName(int c);

CExpr* lexParsePragmaOMP(char *p, int *token);
void out_OMP_PRAGMA(FILE *fp, int indent, int code, CExpr* expr);

#endif
