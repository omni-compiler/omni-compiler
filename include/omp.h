/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#ifndef _OMP_H
#define _OMP_H
typedef void *omp_lock_t;  /* represented as a pointer */
typedef void *omp_nest_lock_t; /* represented as a pointer */

/*
 * Execution Environment Functions
 */
void omp_set_num_threads(int num);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_thread_num(void);
int omp_get_num_procs(void);
int omp_in_parallel(void);
void omp_set_dynamic(int dynamic_thds);
int omp_get_dynamic(void);
void omp_set_nested(int n_nested);
int omp_get_nested(void);
double omp_get_wtime(void);
double omp_get_wtick(void);

/*
 * Lock Functions
 */
void omp_init_lock(omp_lock_t *lock);
void omp_init_nest_lock(omp_nest_lock_t *lock);
void omp_destroy_lock(omp_lock_t *lock);
void omp_destroy_nest_lock(omp_nest_lock_t *lock);
void omp_set_lock(omp_lock_t *lock);
void omp_set_nest_lock(omp_nest_lock_t *lock);
void omp_unset_lock(omp_lock_t *lock);
void omp_unset_nest_lock(omp_nest_lock_t *lock);
int omp_test_lock(omp_lock_t *lock);
int omp_test_nest_lock(omp_nest_lock_t *lock);

#endif /* _OMP_H */
