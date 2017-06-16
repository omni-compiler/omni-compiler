#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "cholesky.h"
#include "tasklet.h"

void task_potrf(void **args);
void task_trsm(void **args);
void task_gemm(void **args);
void task_syrk(void **args);
void cholesky_task(const int ts, const int nt, double* A[nt][nt]);

void cholesky_main();

/* void omp_potrf(double * const A, int ts, int ld) */
void task_potrf(void **args)
{
    static int INFO;
    static const char L = 'L';
    /* dpotrf_(&L, &ts, A, &ld, &INFO); */
    dpotrf_(&L, args[2], args[0], args[2], &INFO);
}

/* void omp_trsm(double *A, double *B, int ts, int ld) */
void task_trsm(void **args)
{
   static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
   static double DONE = 1.0;
   /* dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld ); */
   dtrsm_(&RI, &LO, &TR, &NU, args[2], args[2], &DONE, 
          args[0], args[3], args[1], args[3]);
}

/* void omp_gemm(double *A, double *B, double *C, int ts, int ld) */
void task_gemm(void **args)
{
   static const char TR = 'T', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   /* dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld); */
   dgemm_(&NT, &TR, args[3], args[3], args[3], &DMONE,
          args[0], args[4], args[1], args[4], &DONE, args[2], args[4]);
}

/* void omp_syrk(double *A, double *B, int ts, int ld) */
void task_syrk(void **args)
{
   static char LO = 'L', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   /* dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld ); */
   dsyrk_(&LO, &NT, args[2], args[2], &DMONE,
          args[0], args[3], &DONE,args[1], args[3]);
}

void cholesky_task(const int ts, const int nt, double* A[nt][nt])
{
    for (int k = 0; k < nt; k++) {
        /* #pragma omp task depend(out:A[k][k]) */
        /* omp_potrf (A[k][k], ts, ts); */
        {
            void *args[3];
            void *out_data[1];
            args[0] = (void*) A[k][k]; 
            args[1] = (void*) &ts; 
            args[2] = (void *) &ts;
            out_data[0] = (void *) A[k][k];
            tasklet_create(task_potrf,3,args,0,NULL,1,out_data);
        }

        for (int i = k + 1; i < nt; i++) {
            /* #pragma omp task depend(in:A[k][k]) depend(out:A[k][i]) */
            /* omp_trsm (A[k][k], A[k][i], ts, ts); */
            {
                void *args[4];
                void *in_data[1];
                void *out_data[1];
                args[0] = (void*) A[k][k]; 
                args[1] = (void*) A[k][i]; 
                args[2] = (void*) &ts; 
                args[3] = (void *) &ts;
                in_data[0] = (void*) A[k][k]; 
                out_data[0] = (void*) A[k][i];
                tasklet_create(task_trsm,4,args,1,in_data,1,out_data);
            }
        }

        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {
                /* #pragma omp task depend(in:A[k][i], A[k][j]) depend(out:A[j][i]) */
                /* omp_gemm (A[k][i], A[k][j], A[j][i], ts, ts); */
                {
                    void *args[5];
                    void *in_data[2];
                    void *out_data[1];
                    args[0] = (void*) A[k][i];
                    args[1] = (void*) A[k][j]; 
                    args[2] = (void*) A[j][i]; 
                    args[3] = (void*) &ts; 
                    args[4] = (void *) &ts;
                    in_data[0] = (void*) A[k][i];
                    in_data[1] = (void*) A[k][j];
                    out_data[0] = (void*) A[j][i];
                    tasklet_create(task_gemm,5,args,2,in_data,1,out_data);
                }
            }
            /* #pragma omp task depend(in:A[k][i]) depend(out:A[i][i]) */
            /* omp_syrk (A[k][i], A[i][i], ts, ts); */
            {
                void *args[4];
                void *in_data[1];
                void *out_data[1];
                args[0] = (void*) A[k][i]; 
                args[1] = (void*) A[i][i];
                args[2] = (void*) &ts; 
                args[3] = (void *) &ts;
                in_data[0] = (void*) A[k][i];
                out_data[0] = (void*) A[i][i];
                tasklet_create(task_syrk,4,args,1,in_data,1,out_data);
            }
        }
    }
    // finally, execute it
    tasklet_wait_all();
}

int g_ts;
int g_nt;
double *g_A;

void cholesky_main()
{
    cholesky_task(g_ts, g_nt, (double* (*)[g_nt]) g_A);
}

int main(int argc, char* argv[])
{
    char *result[3] = {"n/a","sucessful","UNSUCCESSFUL"};
    const double eps = BLAS_dfpinfo( blas_eps );

    if (argc < 4) {
        printf( "cholesky matrix_size block_size check\n" );
        exit( -1 );
    }
    const int  n = atoi(argv[1]); // matrix size
    const int ts = atoi(argv[2]); // tile size
    int check    = atoi(argv[3]); // check result?

    double * const matrix = (double *) malloc(n * n * sizeof(double));
    assert(matrix != NULL);

    initialize_matrix(n, ts, matrix);

    double * const original_matrix = (double *) malloc(n * n * sizeof(double));
    assert(original_matrix != NULL);

    const int nt = n / ts;

    double *A[nt][nt];

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            A[i][j] = malloc(ts * ts * sizeof(double));
            assert(A[i][j] != NULL);
        }
    }

    for (int i = 0; i < n * n; i++ ) {
        original_matrix[i] = matrix[i];
    }

    convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, A);

    tasklet_initialize(argc,argv);
    
    /* start */
    const float t1 = get_time();

    /* cholesky_task(ts, nt, (double* (*)[nt]) A); */
    g_ts = ts;
    g_nt = nt;
    g_A = (double *)A;
    tasklet_exec_main(cholesky_main);

    const float t2 = get_time() - t1;
    /* end */
    
    tasklet_finalize();

    convert_to_linear(ts, nt, n, A, (double (*)[n]) matrix);

    if (check) {
        const char uplo = 'L';
        if (check_factorization( n, original_matrix, matrix, n, uplo, eps)) check++;
    }

    free(original_matrix);

    float time = t2;
    float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

    {
      extern int _xmp_num_xstreams;
      printf("test:%s-%d-%d:threads:%2d:result:%s:gflops:%f\n", 
	     argv[0], n, ts, _xmp_num_xstreams, result[check], gflops);
    }

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            assert(A[i][j] != NULL);
            free(A[i][j]);
        }
    }

    free(matrix);
    return 0;
}

