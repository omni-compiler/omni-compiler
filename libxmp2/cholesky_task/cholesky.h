#include <math.h>
// #include <mkl.h>
#include <sys/time.h>
#include <sys/times.h>

void dlacpy_(const char *, int *, int *, double *, int *, double *, int *);
double dlange_(const char *, int *, int *, double *, int *,double *);
void dlarnv_(int *, int *, const int *, double *);
void dpotrf_(const char *, int *, double *, int *, int *);

// #define LAPACK_COL_MAJOR

// #define LAPACKE_dpotrf
#define cblas_dtrsm  dtrsm_
#define cblas_dgemm dgemm_
#define cblas_dsyrk dsyrk_

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
             const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);
void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);
void dtrmm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);
void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
             double *beta, double *c, int *ldc);

enum blas_order_type {
            blas_rowmajor = 101,
            blas_colmajor = 102 };

enum blas_cmach_type {
            blas_base      = 151,
            blas_t         = 152,
            blas_rnd       = 153,
            blas_ieee      = 154,
            blas_emin      = 155,
            blas_emax      = 156,
            blas_eps       = 157,
            blas_prec      = 158,
            blas_underflow = 159,
            blas_overflow  = 160,
            blas_sfmin     = 161};

enum blas_norm_type {
            blas_one_norm       = 171,
            blas_real_one_norm  = 172,
            blas_two_norm       = 173,
            blas_frobenius_norm = 174,
            blas_inf_norm       = 175,
            blas_real_inf_norm  = 176,
            blas_max_norm       = 177,
            blas_real_max_norm  = 178 };

static void BLAS_error(char *rname, int err, int val, int x)
{
	fprintf( stderr, "%s %d %d %d\n", rname, err, val, x );
	abort();
}

static void BLAS_ge_norm(enum blas_order_type order, enum blas_norm_type norm,
		const int m, const int n, const double *a, const int lda, double *res)
{
	char rname[] = "BLAS_ge_norm";

	if (order != blas_colmajor) BLAS_error( rname, -1, order, 0 );

	float anorm, v;
	if (norm == blas_frobenius_norm) {
		anorm = 0.0f;
		for (int j = n; j; --j) {
			for (int i = m; i; --i) {
				v = a[0];
				anorm += v * v;
				a++;
			}
			a += lda - m;
		}
		anorm = sqrt( anorm );
	} else if (norm == blas_inf_norm) {
		anorm = 0.0f;
		for (int i = 0; i < m; ++i) {
			v = 0.0f;
			for (int j = 0; j < n; ++j) {
				v += abs( a[i + j * lda] );
			}
			if (v > anorm)
				anorm = v;
		}
	} else {
		BLAS_error( rname, -2, norm, 0 );
		return;
	}

	if (res) *res = anorm;
}

static double BLAS_dpow_di(double x, int n)
{
	double rv = 1.0;

	if (n < 0) {
		n = -n;
		x = 1.0 / x;
	}

	for (; n; n >>= 1, x *= x) {
		if (n & 1)
			rv *= x;
	}

	return rv;
}

static double BLAS_dfpinfo(enum blas_cmach_type cmach)
{
	const double b = 2.0;
	const int t = 53, l = 1024, m = -1021;
	char rname[] = "BLAS_dfpinfo";

	// for (i = 0; i < t; ++i) eps *= half;
	const double eps = BLAS_dpow_di( b, -t );
	// for (i = 0; i >= m; --i) r *= half;
	const double r = BLAS_dpow_di( b, m-1 );

	double o = 1.0; 
	o -= eps;
	// for (i = 0; i < l; ++i) o *= b;
	o = (o * BLAS_dpow_di( b, l-1 )) * b;

	switch (cmach) {
		case blas_eps: return eps;
		case blas_sfmin: return r;
		default:
			BLAS_error( rname, -1, cmach, 0 );
			break;
	}
	return 0.0;
}

void add_to_diag_hierarchical (double ** matrix, const int ts, const int nt, const float alpha)
{
	for (int i = 0; i < nt * ts; i++)
		matrix[(i/ts) * nt + (i/ts)][(i%ts) * ts + (i%ts)] += alpha;
}

void add_to_diag(double * matrix, const int n, const double alpha)
{
	for (int i = 0; i < n; i++)
		matrix[ i + i * n ] += alpha;
}

float get_time()
{
	static double gtod_ref_time_sec = 0.0;

	struct timeval tv;
	gettimeofday(&tv, NULL);

	// If this is the first invocation of through dclock(), then initialize the
	// "reference time" global variable to the seconds field of the tv struct.
	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double) tv.tv_sec;

	// Normalize the seconds field of the tv struct so that it is relative to the
	// "reference time" that was recorded during the first invocation of dclock().
	const double norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

	// Compute the number of seconds since the reference time.
	const double t = norm_sec + tv.tv_usec * 1.0e-6;

	return (float) t;
}

// Robust Check the factorization of the matrix A2
static int check_factorization(int N, double *A1, double *A2, int LDA, char uplo, double eps)
{
	char NORM = 'I', ALL = 'A', UP = 'U', LO = 'L', TR = 'T', NU = 'N', RI = 'R';

#ifdef VERBOSE
	printf ("Checking result ...\n");
#endif

	double *Residual = (double *)malloc(N*N*sizeof(double));
	double *L1       = (double *)malloc(N*N*sizeof(double));
	double *L2       = (double *)malloc(N*N*sizeof(double));
	double *work     = (double *)malloc(N*sizeof(double));

	memset((void*)L1, 0, N*N*sizeof(double));
	memset((void*)L2, 0, N*N*sizeof(double));

	double alpha= 1.0;

	dlacpy_(&ALL, &N, &N, A1, &LDA, Residual, &N);

	/* Dealing with L'L or U'U  */
	if (uplo == 'U'){
		dlacpy_(&UP, &N, &N, A2, &LDA, L1, &N);
		dlacpy_(&UP, &N, &N, A2, &LDA, L2, &N);
		dtrmm_(&LO, &uplo, &TR, &NU, &N, &N, &alpha, L1, &N, L2, &N);
	}
	else{
		dlacpy_(&LO, &N, &N, A2, &LDA, L1, &N);
		dlacpy_(&LO, &N, &N, A2, &LDA, L2, &N);
		dtrmm_(&RI, &LO, &TR, &NU, &N, &N, &alpha, L1, &N, L2, &N);
	}

	/* Compute the Residual || A -L'L|| */
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];

	double Rnorm = dlange_(&NORM, &N, &N, Residual, &N, work);
	double Anorm = dlange_(&NORM, &N, &N, A1, &N, work);

#ifdef VERBOSE
	printf("============\n");
	printf("Checking the Cholesky Factorization \n");
	printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
#endif

	const int info_factorization = isnan(Rnorm/(Anorm*N*eps)) ||
								   isinf(Rnorm/(Anorm*N*eps)) || 
								   (Rnorm/(Anorm*N*eps) > 60.0);

#ifdef VERBOSE
	if ( info_factorization){
		printf("\n-- Factorization is suspicious ! \n\n");
	}
	else{
		printf("\n-- Factorization is CORRECT ! \n\n");
	}
#endif

	free(Residual); free(L1); free(L2); free(work);

	return info_factorization;
}

void initialize_matrix(const int n, const int ts, double *matrix)
{
	int ISEED[4] = {0,0,0,1};
	int intONE=1;

#ifdef VERBOSE
	printf("Initializing matrix with random values ...\n");
#endif

	for (int i = 0; i < n*n; i+=n) {
		dlarnv_(&intONE, &ISEED[0], &n, &matrix[i]);
	}

	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			matrix[j*n + i] = matrix[j*n + i] + matrix[i*n + j];
			matrix[i*n + j] = matrix[j*n + i];
		}
	}

	add_to_diag(matrix, n, (double) n);
}

static void gather_block(const int N, const int ts, double *Alin, double *A)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			A[i*ts + j] = Alin[i*N + j];
		}
}

static void scatter_block(const int N, const int ts, double *A, double *Alin)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			Alin[i*N + j] = A[i*ts + j];
		}
}

static void convert_to_blocks(const int ts, const int DIM, const int N, double Alin[N][N], double *A[DIM][DIM])
{
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++) {
			gather_block ( N, ts, &Alin[i*ts][j*ts], A[i][j]);
		}
}
static void convert_to_blocks_per_rank(const int ts, const int DIM, const int N, double Alin[N][N], double *A[DIM][DIM], 
                                       int *block_rank, int mype)
{
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++) {
            if (block_rank[i*DIM+j] == mype)
			    gather_block ( N, ts, &Alin[i*ts][j*ts], A[i][j]);
		}
}

static void convert_to_linear(const int ts, const int DIM, const int N, double *A[DIM][DIM], double Alin[N][N])
{
	for (int i = 0; i < DIM; i++)
		for (int j = 0; j < DIM; j++) {
			scatter_block ( N, ts, A[i][j], (double *) &Alin[i*ts][j*ts]);
		}
}

static double * malloc_block (const int ts)
{
	double * const block = (double *) malloc(ts * ts * sizeof(double));
    assert(block != NULL);

	return block;
}



