/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 *
 * @file ompf_runtime.c
 */

#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include "config.h"
#include "ompclib.h"

#define PROGRAM_NAME "OMPF"
#define UBOFS_F 1

#define COMP_VENDOR_GNU       'G'
#define COMP_VENDOR_INTEL     'I'

#if (SIZEOF_VOID_P == 4)
#   define XFMT "%08x"
#   define LFMT "%d"
#else
#   define XFMT "%016lx"
#   define LFMT "%ld"
#endif


/* --- GNU Fortran Definitions --- */

typedef ssize_t index_type;

/**
 * @brief gfortran's array dimension descriptor
 */

typedef struct gnuf_array_dim {
    index_type stride;
    index_type lbound;
    index_type ubound;
} gnuf_array_dim;

/**
 * @brief gfortran's array descriptor
 */

typedef struct gnuf_array {
    char *data;
    size_t offset;
    index_type dtype;
    gnuf_array_dim dims[];
} gnuf_array;


/* --- Intel Fortran Definitions --- */

#if (SIZEOF_VOID_P == 4)
typedef int32_t intelf_int;
#elif (SIZEOF_VOID_P == 8)
typedef int64_t intelf_int;
#else
#error unsupported pointer size
#endif

/**
 * intel fortran's array dimension descriptor
 */

typedef struct intelf_array_dim {
    intelf_int size;
    intelf_int step_bytes;
    intelf_int lbound;
} intelf_array_dim;

/**
 * intel fortran's array descriptor
 */

typedef struct intelf_array {
    char *data;
    intelf_int elem_size;
    intelf_int offset;
    intelf_int allocated;
    intelf_int ndims;
    intelf_int reserved;
    intelf_array_dim dims[];
} intelf_array;


/**
 * runtime functions
 */

void
ompf_init_()
{
    char *argv[1];
    argv[0] = (char*)malloc(8);
    strcpy(argv[0], PROGRAM_NAME);
    ompc_init(1, argv);
}


void
ompf_terminate_()
{
    ompc_terminate(0);
}


void
ompf_do_parallel_(int *pnargs, int *cond, cfunc f, ...)
{
    int i;
    int nargs = *pnargs;
    void *argv[nargs];
    va_list vargs;

    va_start(vargs, f);
    for(i = 0; i < nargs; ++i) {
        argv[i] = va_arg(vargs, void*);
    }
    va_end(vargs);
    ompc_do_parallel_main(nargs, *cond, ompc_num_threads, f, (void*)argv);
}


void
ompf_barrier_()
{
    extern void ompc_barrier();
    ompc_barrier();
}


void
ompf_current_thread_barrier_()
{
    extern void ompc_current_thread_barrier();
    ompc_current_thread_barrier();
}


void
ompf_default_sched_(indvar_t *lb, indvar_t *ub, int *step)
{
    extern void _ompc_default_sched();
    _ompc_default_sched(lb, ub, UBOFS_F, step);
}


void
ompf_static_csched_(indvar_t *lb, indvar_t *ub, int *step)
{
    extern void _ompc_static_csched();
    _ompc_static_csched(lb, ub, UBOFS_F, step);
}


void
ompf_static_bsched_(indvar_t *lb, indvar_t *ub, int *step)
{
    extern void _ompc_static_bsched();
    _ompc_static_bsched(lb, ub, UBOFS_F, step);
}


void
ompf_static_sched_init_(indvar_t *lb, indvar_t *ub, int *step,
    int *chunk_size)
{
    extern void _ompc_static_sched_init();
    _ompc_static_sched_init(*lb, *ub, UBOFS_F, *step, *chunk_size);
}


int
ompf_static_sched_next_(indvar_t *lb, indvar_t *ub)
{
    extern int _ompc_static_sched_next();
    return _ompc_static_sched_next(lb, ub, UBOFS_F);
}


void
ompf_dynamic_sched_init_(
    indvar_t *lb, indvar_t *ub, int *step, int *chunk_size)
{
    extern void _ompc_dynamic_sched_init();
    _ompc_dynamic_sched_init(*lb, *ub, UBOFS_F, *step, *chunk_size);
}


int
ompf_dynamic_sched_next_(indvar_t *lb, indvar_t *ub)
{
    extern int _ompc_dynamic_sched_next();
    return _ompc_dynamic_sched_next(lb, ub, UBOFS_F);
}


void
ompf_guided_sched_init_(indvar_t *lb, indvar_t *ub, int *step,
    int *chunk_size)
{
    extern void _ompc_guided_sched_init();
    _ompc_guided_sched_init(*lb, *ub, UBOFS_F, *step, *chunk_size);
}


int
ompf_guided_sched_next_(indvar_t *lb, indvar_t *ub)
{
    extern int _ompc_guided_sched_next();
    return _ompc_guided_sched_next(lb, ub, UBOFS_F);
}


void
ompf_runtime_sched_init_(indvar_t *lb, indvar_t *ub, int *step)
{
    extern void _ompc_runtime_sched_init();
    _ompc_runtime_sched_init(*lb, *ub, UBOFS_F, *step);
}


int
ompf_runtime_sched_next_(indvar_t *lb, indvar_t *ub)
{
    extern int _ompc_runtime_sched_next();
    return _ompc_runtime_sched_next(lb, ub, UBOFS_F);
}


void
ompf_set_loop_id_(indvar_t *i)
{
    extern void ompc_set_loop_id();
    ompc_set_loop_id(*i);
}


void
ompf_init_ordered_(indvar_t *lb, int *step)
{
    extern void ompc_init_ordered();
    ompc_init_ordered(*lb, *step);
}


void
ompf_ordered_begin_()
{
    extern void ompc_ordered_begin();
    ompc_ordered_begin();
}


void
ompf_ordered_end_()
{
    extern void ompc_ordered_end();
    ompc_ordered_end();
}


void
ompf_section_init_(int *n_sections)
{
    extern void ompc_section_init();
    ompc_section_init(*n_sections);
}


int
ompf_section_id_()
{
    extern int ompc_section_id();
    return ompc_section_id();
}


int
ompf_is_last_()
{
    extern int ompc_is_last();
    return ompc_is_last();
}


int
ompf_do_single_()
{
    extern int ompc_do_single();
    return ompc_do_single();
}


int
ompf_is_master_()
{
    extern int ompc_is_master();
    return ompc_is_master();
}


void
ompf_enter_critical_(ompc_lock_t **lock)
{
    ompc_enter_critical(lock);
}


void
ompf_exit_critical_(ompc_lock_t **lock)
{
    ompc_exit_critical(lock);
}


void
ompf_atomic_init_lock_()
{
    extern void ompc_atomic_init_lock();
    ompc_atomic_init_lock();
}


void
ompf_atomic_lock_()
{
    extern void ompc_atomic_lock();
    ompc_atomic_lock();
}


void
ompf_atomic_unlock_()
{
    extern void ompc_atomic_unlock();
    ompc_atomic_unlock();
}


void
ompf_thread_lock_()
{
    extern void ompc_thread_lock();
    ompc_thread_lock();
}


void
ompf_thread_unlock_()
{
    extern void ompc_thread_unlock();
    ompc_thread_unlock();
}


void
ompf_atomic_destroy_lock_()
{
    extern void ompc_atomic_destroy_lock();
    ompc_atomic_destroy_lock();
}


void
ompf_flush_()
{
    extern void ompc_flush();
    ompc_flush();
}


int
ompf_get_num_threads_()
{
    extern int omp_get_num_threads();
    return omp_get_num_threads();
}


void
ompf_set_num_threads_(int *n)
{
    extern void ompc_set_num_threads();
    ompc_set_num_threads(*n);
}


int
ompf_get_thread_num_()
{
    extern int ompc_get_thread_num();
    return ompc_get_thread_num();
}


int
ompf_get_max_threads_()
{
    extern int ompc_get_max_threads();
    return ompc_get_max_threads();
}


uintptr_t
ompf_get_addr_(void *p)
{
    return (uintptr_t)p;
}


/** @brief set array bounds (lower bound / upper bound / base addr offset) */
void
ompf_set_bounds(int compiler_vendor, void *src, void *dst, int ndims, ...)
{
    int i;
    va_list vargs;

    if(src == dst)
        return;

    va_start(vargs, ndims);

    switch(compiler_vendor) {
    case COMP_VENDOR_GNU:
        ((gnuf_array*)dst)->offset = ((gnuf_array*)src)->offset;
        break;
    case COMP_VENDOR_INTEL:
        ((intelf_array*)dst)->offset = ((intelf_array*)src)->offset;
        break;
    }

    for(i = 0; i < ndims; ++i) {
        int lb = *va_arg(vargs, int*);

        switch(compiler_vendor) {
        case COMP_VENDOR_GNU: {
            gnuf_array *a = (gnuf_array*)dst;
            gnuf_array_dim *d = &a->dims[i];
            d->ubound = d->ubound - d->lbound + lb;
            d->lbound = lb;
        }
            break;
        case COMP_VENDOR_INTEL: {
            intelf_array *a = (intelf_array*)dst;
            intelf_array_dim *d = &a->dims[i];
            d->lbound = lb;
        }
            break;
        default:
            perror("invalid compiler type in ompf_set_bounds");
            exit(1);
        }
    }

    va_end(vargs);
}


/** @brief 1 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_1_(int *compiler_vendor, int *src, int *dst,
    int *a1)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 1,
        a1);
}


/** @brief 2 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_2_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 2,
        a1, a2);
}


/** @brief 3 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_3_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2, int *a3)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 3,
        a1, a2, a3);
}


/** @brief 4 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_4_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2, int *a3, int *a4)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 4,
        a1, a2, a3, a4);
}


/** @brief 5 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_5_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2, int *a3, int *a4, int *a5)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 5,
        a1, a2, a3, a4, a5);
}


/** @brief 6 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_6_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2, int *a3, int *a4, int *a5, int *a6)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 6,
        a1, a2, a3, a4, a5, a6);
}

/** @brief 7 dimension version of ompf_set_bounds_#_ */
void
ompf_set_bounds_7_(int *compiler_vendor, int *src, int *dst,
    int *a1, int *a2, int *a3, int *a4, int *a5, int *a6, int *a7)
{
    ompf_set_bounds(*compiler_vendor, src, dst, 7,
        a1, a2, a3, a4, a5, a6, a7);
}


/**
 * @brief get POINTER's reference addr.
 * use following interface declaration.
 *
 * interface
 *     function ompf_get_ref_addr(p)
 *         integer::ompf_get_ref_addr
 *         [type],pointer::p
 *     end function
 * end interface
 */
uintptr_t
ompf_get_ref_addr_(void **p)
{
    return (uintptr_t)*p;
}


int
ompf_master_()
{
    return ompf_is_master_();
}


void
ompf_debug_tn_(char *msg, int len)
{
    extern int omp_get_thread_num();
    char buf[len + 1];
    memcpy(buf, msg, len);
    buf[len] = 0;
    printf("[%d] : %s\n", omp_get_thread_num(), buf);
    fflush(stdout);
}


void
ompf_debug_tn_i_(char *msg, int *val, int len)
{
    extern int omp_get_thread_num();
    char buf[len + 1];
    memcpy(buf, msg, len);
    buf[len] = 0;
    printf("[%d] : %s val=%d\n", omp_get_thread_num(), buf, *val);
    fflush(stdout);
}


void
ompf_debug_tn_a_(char *msg, int *val, int len)
{
    extern int omp_get_thread_num();
    char buf[len + 1];
    memcpy(buf, msg, len);
    buf[len] = 0;
    printf("[%d] : %s val=0x" XFMT "\n", omp_get_thread_num(), buf, (uintptr_t)val);
    fflush(stdout);
}


/**
 * @brief get size of array header.
 */
static int
get_size_of_array_header(int compiler_vendor, int ndims)
{
    switch(compiler_vendor) {
    case COMP_VENDOR_GNU:
        return sizeof(gnuf_array) + ndims * sizeof(gnuf_array_dim);
    case COMP_VENDOR_INTEL:
        return sizeof(intelf_array) + ndims * sizeof(intelf_array_dim);
    }
    perror("invalid compiler_vendor");
    exit(1);
    return 0;
}


/**
 * @brief copy pointer value from src to dst.
 */
void
ompf_save_array_header_(int *compiler_vendor, void *src, void **dst)
{
    *dst = src;

#if 0
    printf("#save: "XFMT" <- "XFMT"\n", (uintptr_t)dst, (uintptr_t)src);
    gnuf_array *a = (gnuf_array*)src;
    printf("  data="XFMT" offset="LFMT" dtype="LFMT"\n", (uintptr_t)a->data, a->offset, a->dtype);
    fflush(stdout);
#endif
}


/**
 * @brief copy array header from src to dst.
 */
void
ompf_load_array_header_(int *compiler_vendor, void **src, void *dst, int *pndims)
{
    int sz = get_size_of_array_header(*compiler_vendor, *pndims);
    memcpy(dst, *src, sz);
#if 0
    printf("#load: "XFMT" <- "XFMT" sz=%d ndims=%d\n", (uintptr_t)dst, (uintptr_t)src, sz, *pndims);
    gnuf_array *a = (gnuf_array*)src;
    printf("  data="XFMT" offset="LFMT" dtype="LFMT"\n", (uintptr_t)a->data, a->offset, a->dtype);
    fflush(stdout);
#endif
}

