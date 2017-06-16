#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "omp.h"

#include "cholesky.h"

#if 0
void omp_potrf(double * const A, int ts, int ld)
{
    static int INFO;
    static const char L = 'L';
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, L, ts, A, ts);
}
void omp_trsm(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    static double DONE = 1.0;
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, ts, ts, DONE, A, ld, B, ld);
}
void omp_gemm(double *A, double *B, double *C, int ts, int ld)
{
    static const char TR = 'T', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ts, ts, ts, DMONE, A, ld, B, ld, DONE, C, ld);
}
void omp_syrk(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, ts, ts, DMONE, A, ld, DONE, B, ld);
}

#else

void omp_potrf(double * const A, int ts, int ld)
{
   static int INFO;
   static const char L = 'L';
   dpotrf_(&L, &ts, A, &ld, &INFO);
}

void omp_trsm(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
   static double DONE = 1.0;
   dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

void omp_gemm(double *A, double *B, double *C, int ts, int ld)
{
   static const char TR = 'T', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}

void omp_syrk(double *A, double *B, int ts, int ld)
{
   static char LO = 'L', NT = 'N';
   static double DONE = 1.0, DMONE = -1.0;
   dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}
#endif

#if 0
void cholesky(const int ts, const int nt, double* A[nt][nt])
{
#pragma omp parallel
#pragma omp single
    for (int k = 0; k < nt; k++) {
#pragma omp task depend(out:A[k][k])
{
        omp_potrf(A[k][k], ts, ts);
}
        for (int i = k + 1; i < nt; i++) {
#pragma omp task depend(in:A[k][k]) depend(out:A[k][i])
{
            omp_trsm(A[k][k], A[k][i], ts, ts);
}
        }
        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {
#pragma omp task depend(in:A[k][i], A[k][j]) depend(out:A[j][i])
{
                omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
}
            }
#pragma omp task depend(in:A[k][i]) depend(out:A[i][i])
{
            omp_syrk(A[k][i], A[i][i], ts, ts);
}
        }
    }
#pragma omp taskwait
}

#else

void cholesky(const int ts, const int nt, double* A[nt][nt])
{
    for (int k = 0; k < nt; k++) {
        omp_potrf(A[k][k], ts, ts);
        for (int i = k + 1; i < nt; i++) {
            omp_trsm(A[k][k], A[k][i], ts, ts);
        }
        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {
                omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
            }
            omp_syrk(A[k][i], A[i][i], ts, ts);
        }
    }
}
#endif

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

    const float t1 = get_time();
    cholesky(ts, nt, (double* (*)[nt]) A);
    const float t2 = get_time() - t1;

    convert_to_linear(ts, nt, n, A, (double (*)[n]) matrix);

    if (check) {
        const char uplo = 'L';
        if (check_factorization( n, original_matrix, matrix, n, uplo, eps)) check++;
    }

    free(original_matrix);

    float time = t2;
    float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

#if 0
#pragma omp parallel
#pragma omp single
    printf("test:%s-%d-%d:threads:%2d:result:%s:gflops:%f\n", argv[0], n, ts, omp_get_num_threads(), result[check], gflops );
#else
    printf("test:%s-%d-%d:threads:%2d:result:%s:gflops:%f\n", argv[0], n, ts, 0, result[check], gflops );
#endif

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            assert(A[i][j] != NULL);
            free(A[i][j]);
        }
    }

    free(matrix);
    return 0;
}

