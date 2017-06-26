#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <mpi.h>

#include "cholesky.h"
#include "tasklet.h"

static int mype, np;

void task_potrf(void **args);
void task_trsm(void **args);
void task_gemm(void **args);
void task_syrk(void **args);
void cholesky_task(const int ts, const int nt, double* A[nt][nt], double* B, double* C[nt], int* block_rank);
int verify(int ts, int nt, double *A[nt][nt], double *original_matrix, int *block_rank, int np, int mype);

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

/* void omp_send(double *A, int size, int dst, int tag); */
void task_send(void **args)
{
    int comp = 0;
    MPI_Request req;
    MPI_Isend(args[0], (int)(intptr_t)args[1], MPI_DOUBLE, (int)(intptr_t)args[2], 
              (int)(intptr_t)args[3], MPI_COMM_WORLD, &req);

    while (!comp) {
        tasklet_yield();
        MPI_Test(&req, &comp, MPI_STATUS_IGNORE);
    }
}

/* void omp_recv(double *A, int size, int src, int tag); */
void task_recv(void **args)
{
    int comp = 0;
    MPI_Request req;
    MPI_Irecv(args[0], (int)(intptr_t)args[1], MPI_DOUBLE, (int)(intptr_t)args[2],
              (int)(intptr_t)args[3], MPI_COMM_WORLD, &req);

    while (!comp) {
        tasklet_yield();
        MPI_Test(&req, &comp, MPI_STATUS_IGNORE);
    }
}

void cholesky_task(const int ts, const int nt, double* A[nt][nt], double* B, double* C[nt], int* block_rank)
{
    char *send_flags = (char *) malloc(np * sizeof(char)), recv_flag;

    for (int k = 0; k < nt; k++) {

        if (block_rank[k*nt+k] == mype) {
            /* #pragma omp task depend(out:A[k][k]) */
            /* omp_potrf (A[k][k], ts, ts); */
            {
                void *args[3];
                void *out_data[1];
                args[0] = (void*) A[k][k]; 
                args[1] = (void*) &ts; 
                args[2] = (void*) &ts;
                out_data[0] = (void*) A[k][k];
                tasklet_create(task_potrf,3,args,0,NULL,1,out_data);
            }
        }

        if (block_rank[k*nt+k] == mype) {

            memset((void *) send_flags, FALSE, np * sizeof(char));
            for (int kk = k + 1; kk < nt; kk++)
                if (send_flags[block_rank[k*nt+kk]] == FALSE)
                    send_flags[block_rank[k*nt+kk]] = TRUE;

            for (int dst = 0; dst < np; dst++) {
                if (send_flags[dst] && dst != mype) {
                    int size = ts*ts;
                    int tag = k*nt+k;
                    /* #pragma omp task depend(in:A[k][k]) */
                    /* omp_send (A[k][k], size, dst, tag); */
                    {
                        void *args[4];
                        void *in_data[1];
                        args[0] = (void*) A[k][k];
                        args[1] = (void*) size;
                        args[2] = (void*) dst;
                        args[3] = (void*) tag;
                        in_data[0] = (void*) A[k][k];
                        tasklet_create(task_send,4,args,1,in_data,0,NULL);
                    }
                }
            }
        }
        if (block_rank[k*nt+k] != mype) {

            recv_flag = FALSE;
            for (int kk = k + 1; kk < nt; kk++)
                if (block_rank[k*nt+kk] == mype)
                    recv_flag = TRUE;

            if (recv_flag) {
                int size = ts*ts;
                int tag = k*nt+k;
                /* #pragma omp task depend(out:B) */
                /* omp_recv (B, size, src, tag); */
                {
                    void *args[4];
                    void *out_data[1];
                    args[0] = (void*) B;
                    args[1] = (void*) size;
                    args[2] = (void*) block_rank[k*nt+k];
                    args[3] = (void*) tag;
                    out_data[0] = (void*) B;
                    tasklet_create(task_recv,4,args,0,NULL,1,out_data);
                }
            }
        }

        for (int i = k + 1; i < nt; i++) {
            
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
                    /* #pragma omp task depend(in:A[k][k]) depend(out:A[k][i]) */
                    /* omp_trsm (A[k][k], A[k][i], ts, ts); */
                    {
                        void *args[4];
                        void *in_data[1];
                        void *out_data[1];
                        args[0] = (void*) A[k][k]; 
                        args[1] = (void*) A[k][i]; 
                        args[2] = (void*) &ts; 
                        args[3] = (void*) &ts;
                        in_data[0] = (void*) A[k][k]; 
                        out_data[0] = (void*) A[k][i];
                        tasklet_create(task_trsm,4,args,1,in_data,1,out_data);
                    }
                } else {
                    /* #pragma omp task depend(in:B) depend(out:A[k][i]) */
                    /* omp_trsm (B, A[k][i], ts, ts); */
                    {
                        void *args[4];
                        void *in_data[1];
                        void *out_data[1];
                        args[0] = (void*) B; 
                        args[1] = (void*) A[k][i]; 
                        args[2] = (void*) &ts; 
                        args[3] = (void*) &ts;
                        in_data[0] = (void*) B; 
                        out_data[0] = (void*) A[k][i];
                        tasklet_create(task_trsm,4,args,1,in_data,1,out_data);
                    }
                }
            }

            if (block_rank[k*nt+i] == mype) {

                memset((void *) send_flags, FALSE, np * sizeof(char));
                for (int ii = k + 1; ii < i; ii++) 
                    if (send_flags[block_rank[ii*nt+i]] == FALSE)
                        send_flags[block_rank[ii*nt+i]] = TRUE;
                for (int ii = i + 1; ii < nt; ii++)
                    if (send_flags[block_rank[i*nt+ii]] == FALSE)
                        send_flags[block_rank[i*nt+ii]] = TRUE;
                if (send_flags[block_rank[i*nt+i]] == FALSE)
                    send_flags[block_rank[i*nt+i]] = TRUE;

                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                        int size = ts*ts;
                        int tag = k*nt+i;
                        /* #pragma omp task depend(in:A[k][i]) */
                        /* omp_send (A[k][i]); */
                        {
                            void *args[4];
                            void *in_data[1];
                            args[0] = (void*) A[k][i];
                            args[1] = (void*) size;
                            args[2] = (void*) dst;
                            args[3] = (void*) tag;
                            in_data[0] = (void*) A[k][i];
                            tasklet_create(task_send,4,args,1,in_data,0,NULL);
                        }
                    }
                }
            }

            if (block_rank[k*nt+i] != mype) {

                recv_flag = FALSE;
                for (int ii = k + 1; ii < i; ii++) 
                    if (block_rank[ii*nt+i] == mype)
                        recv_flag = TRUE;
                for (int ii = i + 1; ii < nt; ii++)
                    if (block_rank[i*nt+ii] == mype)
                        recv_flag = TRUE;
                if (block_rank[i*nt+i] == mype)
                    recv_flag = TRUE;

                if (recv_flag) {
                    int size = ts*ts;
                    int tag = k*nt+i;
                    /* #pragma omp task depend(out:C[i]) */
                    /* omp_recv (C[i]) */
                    {
                        void *args[4];
                        void *out_data[1];
                        args[0] = (void*) C[i];
                        args[1] = (void*) size;
                        args[2] = (void*) block_rank[k*nt+i];
                        args[3] = (void*) tag;
                        out_data[0] = (void*) C[i];
                        tasklet_create(task_recv,4,args,0,NULL,1,out_data);
                    }
                }
            }

        }

        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {

                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
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
                            args[4] = (void*) &ts;
                            in_data[0] = (void*) A[k][i];
                            in_data[1] = (void*) A[k][j];
                            out_data[0] = (void*) A[j][i];
                            tasklet_create(task_gemm,5,args,2,in_data,1,out_data);
                        }
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
                        /* #pragma omp task depend(in:C[i], A[k][j]) depend(out:A[j][i]) */
                        /* omp_gemm (C[i], A[k][j], A[j][i], ts, ts); */
                        { 
                            void *args[5];
                            void *in_data[2];
                            void *out_data[1];
                            args[0] = (void*) C[i];
                            args[1] = (void*) A[k][j]; 
                            args[2] = (void*) A[j][i]; 
                            args[3] = (void*) &ts; 
                            args[4] = (void*) &ts;
                            in_data[0] = (void*) C[i];
                            in_data[1] = (void*) A[k][j];
                            out_data[0] = (void*) A[j][i];
                            tasklet_create(task_gemm,5,args,2,in_data,1,out_data);
                        }
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                        /* #pragma omp task depend(in:A[k][i], C[j]) depend(out:A[j][i]) */
                        /* omp_gemm (A[k][i], C[j], A[j][i], ts, ts); */
                        { 
                            void *args[5];
                            void *in_data[2];
                            void *out_data[1];
                            args[0] = (void*) A[k][i];
                            args[1] = (void*) C[j]; 
                            args[2] = (void*) A[j][i]; 
                            args[3] = (void*) &ts; 
                            args[4] = (void*) &ts;
                            in_data[0] = (void*) A[k][i];
                            in_data[1] = (void*) C[j];
                            out_data[0] = (void*) A[j][i];
                            tasklet_create(task_gemm,5,args,2,in_data,1,out_data);
                        }
                    } else {
                        /* #pragma omp task depend(in:C[i], C[j]) depend(out:A[j][i]) */
                        /* omp_gemm (C[i], C[j], A[j][i], ts, ts); */
                        { 
                            void *args[5];
                            void *in_data[2];
                            void *out_data[1];
                            args[0] = (void*) C[i];
                            args[1] = (void*) C[j]; 
                            args[2] = (void*) A[j][i]; 
                            args[3] = (void*) &ts; 
                            args[4] = (void*) &ts;
                            in_data[0] = (void*) C[i];
                            in_data[1] = (void*) C[j];
                            out_data[0] = (void*) A[j][i];
                            tasklet_create(task_gemm,5,args,2,in_data,1,out_data);
                        }
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
                    /* #pragma omp task depend(in:A[k][i]) depend(out:A[i][i]) */
                    /* omp_syrk (A[k][i], A[i][i], ts, ts); */
                    {
                        void *args[4];
                        void *in_data[1];
                        void *out_data[1];
                        args[0] = (void*) A[k][i]; 
                        args[1] = (void*) A[i][i];
                        args[2] = (void*) &ts; 
                        args[3] = (void*) &ts;
                        in_data[0] = (void*) A[k][i];
                        out_data[0] = (void*) A[i][i];
                        tasklet_create(task_syrk,4,args,1,in_data,1,out_data);
                    }
                } else {
                    /* #pragma omp task depend(in:C[i]) depend(out:A[i][i]) */
                    /* omp_syrk (C[i], A[i][i], ts, ts); */
                    {
                        void *args[4];
                        void *in_data[1];
                        void *out_data[1];
                        args[0] = (void*) C[i]; 
                        args[1] = (void*) A[i][i];
                        args[2] = (void*) &ts; 
                        args[3] = (void*) &ts;
                        in_data[0] = (void*) C[i];
                        out_data[0] = (void*) A[i][i];
                        tasklet_create(task_syrk,4,args,1,in_data,1,out_data);
                    }
                }
            }
        }
    }
    // finally, execute it
    tasklet_wait_all();
    MPI_Barrier(MPI_COMM_WORLD);

    free(send_flags);
}

int g_ts;
int g_nt;
double *g_A;
double *g_B;
double *g_C;
int *g_block_rank;

void cholesky_main()
{
    cholesky_task(g_ts, g_nt, (double* (*)[g_nt]) g_A, g_B, (double**) g_C, g_block_rank);
}

/* 2D block-cyclic distribution */
void get_block_rank(int nt, int *block_rank)
{
    int i, j, row, col, tmp_rank = 0, offset = 0;
    row = col = np;

    if (np != 1) {
        while (1) {
            row = row / 2;
            if (row * col == np) break;
            col = col / 2;
            if (row * col == np) break;
        }
    }
    for (i = 0; i < nt; i++) {
        for (j = 0; j < nt; j++) {
            block_rank[i*nt+j] = tmp_rank + offset;
            tmp_rank++;
            if (tmp_rank >= col) tmp_rank = 0;
        }
        tmp_rank = 0;
        offset = (offset + col >= np) ? 0 : offset + col;
    }
}

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    char *result[3] = {"n/a","sucessful","UNSUCCESSFUL"};

    if (argc < 4) {
        printf( "cholesky matrix_size block_size check\n" );
        exit( -1 );
    }
    const int  n = atoi(argv[1]); // matrix size
    const int ts = atoi(argv[2]); // tile size
    int check    = atoi(argv[3]); // check result?

    double * const original_matrix = (double *) malloc(n * n * sizeof(double));
    assert(original_matrix != NULL);

    initialize_matrix(n, ts, original_matrix);

    const int nt = n / ts;

    int *block_rank = (int *) malloc(nt * nt * sizeof(int));
    assert(block_rank != NULL);

    get_block_rank(nt, block_rank);

    double *A[nt][nt], *B, *C[nt];

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            if (block_rank[i*nt+j] == mype) {
                A[i][j] = (double *) malloc(ts * ts * sizeof(double));
                assert(A[i][j] != NULL);
            }
        }
        C[i] = (double *) malloc(ts * ts * sizeof(double));
        assert(C[i] != NULL);
    }
    B = (double *) malloc(ts * ts * sizeof(double));
    assert(B != NULL);

    convert_to_blocks_per_rank(ts, nt, n, (double(*)[n]) original_matrix, A, block_rank, mype);

    tasklet_initialize(argc,argv);

    MPI_Barrier(MPI_COMM_WORLD);
 
    /* start */
    const float t1 = get_time();

    /* cholesky_task(ts, nt, (double* (*)[nt]) A, B, (double**) C, block_rank); */
    g_ts = ts;
    g_nt = nt;
    g_A = (double *)A;
    g_B = B;
    g_C = (double *)C;
    g_block_rank = block_rank;
    tasklet_exec_main((cfunc)cholesky_main);

    const float t2 = get_time() - t1;
    /* end */

    MPI_Barrier(MPI_COMM_WORLD);   
 
    tasklet_finalize();

    /* Verification */
    if (check != 0)
        check = verify(ts, nt, A, original_matrix, block_rank, np, mype);

    float time = t2;
    float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

    {
      extern int _xmp_num_xstreams;
      printf("test:%s-%d-%d:np:%2d:mype:%2d:threads:%2d:result:%s:gflops:%f:time:%f\n", 
	     argv[0], n, ts, np, mype, _xmp_num_xstreams, result[check], gflops, t2);
    }

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            if (block_rank[i*nt+j] == mype) {
                free(A[i][j]);
            }
        }
        free(C[i]);
    }
    free(B);
    free(block_rank);
    free(original_matrix);

    MPI_Finalize();

    return 0;
}

int verify(int ts, int nt, double *A[nt][nt], double *original_matrix, int *block_rank, int np, int mype)
{
    int val;
    double *tmp_A[nt][nt];
    double *matrix = (double *) malloc(nt*nt*ts*ts*sizeof(double));

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            if (mype == 0) {
                tmp_A[i][j] = (double *) malloc(ts*ts*sizeof(double));
                if (block_rank[i*nt+j] != mype)
                    MPI_Recv(tmp_A[i][j], ts*ts, MPI_DOUBLE, block_rank[i*nt+j], i*nt+j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                else
                    memcpy(tmp_A[i][j], A[i][j], ts*ts*sizeof(double));
            } else {
                if (block_rank[i*nt+j] == mype)
                    MPI_Send(A[i][j], ts*ts, MPI_DOUBLE, 0, i*nt+j, MPI_COMM_WORLD);
            }
        }
    }

    if (mype == 0) {

        const double eps = BLAS_dfpinfo(blas_eps);
        const char uplo = 'L';
        int n = nt * ts;

        convert_to_linear(ts, nt, n, tmp_A, (double (*)[n]) matrix);

        if (check_factorization(n, original_matrix, matrix, n, uplo, eps))
            val = 2; /* unsuccessful */
        else
            val = 1; /* successful */

        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                free(tmp_A[i][j]);
            }
        }
    }

    free(matrix);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return val;
}
