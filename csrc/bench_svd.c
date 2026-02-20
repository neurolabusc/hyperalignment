/*
 * bench_svd.c -- Microbenchmark for SVD timing at typical searchlight size.
 * Compile: cc -O2 bench_svd.c -framework Accelerate -o bench_svd
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
typedef __LAPACK_int lapack_int;

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
	int N = 121;
	int iters = 10000;

	/* Fill with repeatable random data */
	srand(42);
	double *A_orig = (double *)malloc((size_t)N * N * sizeof(double));
	for (int i = 0; i < N * N; i++)
		A_orig[i] = (double)rand() / RAND_MAX - 0.5;

	double *A    = (double *)malloc((size_t)N * N * sizeof(double));
	double *U    = (double *)malloc((size_t)N * N * sizeof(double));
	double *s    = (double *)malloc((size_t)N * sizeof(double));
	double *Vt   = (double *)malloc((size_t)N * N * sizeof(double));
	lapack_int *iwork = (lapack_int *)malloc(8 * (size_t)N * sizeof(lapack_int));

	/* Workspace query */
	lapack_int lN = N, info = 0, lwork = -1;
	double work_query;
	char jobz = 'S';
	dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
	        &work_query, &lwork, iwork, &info);
	lwork = (lapack_int)work_query;
	double *work = (double *)malloc((size_t)lwork * sizeof(double));
	printf("N=%d, lwork=%d (%.1f KB)\n", N, (int)lwork, lwork * 8.0 / 1024);

	/* Warmup */
	for (int i = 0; i < 200; i++) {
		memcpy(A, A_orig, (size_t)N * N * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
		        work, &lwork, iwork, &info);
	}

	/* Time: SVD only (reuse workspace) */
	double t0 = now_sec();
	for (int i = 0; i < iters; i++) {
		memcpy(A, A_orig, (size_t)N * N * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
		        work, &lwork, iwork, &info);
	}
	double t1 = now_sec();
	double ms_reuse = (t1 - t0) / iters * 1000;
	printf("SVD (workspace reuse):   %.3f ms/call  (x19341 = %.1fs)\n",
	       ms_reuse, ms_reuse * 19341 / 1000);

	/* Time: SVD with fresh workspace allocation each time (like old code) */
	t0 = now_sec();
	for (int i = 0; i < iters; i++) {
		memcpy(A, A_orig, (size_t)N * N * sizeof(double));
		/* Fresh workspace query + alloc */
		lapack_int fresh_lwork = -1;
		double fresh_query;
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
		        &fresh_query, &fresh_lwork, iwork, &info);
		fresh_lwork = (lapack_int)fresh_query;
		double *fresh_work = (double *)malloc((size_t)fresh_lwork * sizeof(double));
		memcpy(A, A_orig, (size_t)N * N * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
		        fresh_work, &fresh_lwork, iwork, &info);
		free(fresh_work);
	}
	t1 = now_sec();
	double ms_fresh = (t1 - t0) / iters * 1000;
	printf("SVD (fresh alloc each):  %.3f ms/call  (x19341 = %.1fs)\n",
	       ms_fresh, ms_fresh * 19341 / 1000);

	printf("\nPython scipy reference:  1.060 ms/call  (x19341 = 20.5s)\n");

	free(A_orig); free(A); free(U); free(s); free(Vt);
	free(iwork); free(work);
	return 0;
}
