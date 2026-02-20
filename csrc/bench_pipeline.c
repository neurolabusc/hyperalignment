/*
 * bench_pipeline.c -- Microbenchmark for each step of the searchlight pipeline.
 * Compile: cc -O2 bench_pipeline.c -framework Accelerate -o bench_pipeline
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
	int nt = 1818;   /* timepoints (4 concatenated runs) */
	int nv = 9675;   /* total vertices per hemisphere */
	int sz = 121;    /* searchlight size (median) */
	int iters = 2000;
	int n_sls = 9675; /* total searchlights per hemisphere */

	printf("nt=%d, nv=%d, sz=%d (median), n_sls=%d\n\n", nt, nv, sz, n_sls);

	/* Allocate big source matrices */
	double *X = (double *)malloc((size_t)nt * nv * sizeof(double));
	double *Y = (double *)malloc((size_t)nt * nv * sizeof(double));
	srand(42);
	for (size_t i = 0; i < (size_t)nt * nv; i++) {
		X[i] = (double)rand() / RAND_MAX - 0.5;
		Y[i] = (double)rand() / RAND_MAX - 0.5;
	}

	/* Create a representative searchlight index array */
	int *sl = (int *)malloc((size_t)sz * sizeof(int));
	for (int i = 0; i < sz; i++) sl[i] = i * (nv / sz);

	/* Pre-allocate workspace */
	double *local_X = (double *)malloc((size_t)nt * sz * sizeof(double));
	double *local_Y = (double *)malloc((size_t)nt * sz * sizeof(double));
	double *A       = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *A_orig  = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *U       = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *s       = (double *)malloc((size_t)sz * sizeof(double));
	double *Vt      = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *T_local = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *T_out   = (double *)calloc((size_t)nv * nv, sizeof(double));
	lapack_int *iwork = (lapack_int *)malloc(8 * (size_t)sz * sizeof(lapack_int));

	/* LAPACK workspace query */
	lapack_int lN = sz, info = 0, lwork = -1;
	double work_query;
	char jobz = 'S';
	dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
	        &work_query, &lwork, iwork, &info);
	lwork = (lapack_int)work_query;
	double *work = (double *)malloc((size_t)lwork * sizeof(double));

	double *weights = (double *)malloc((size_t)sz * sizeof(double));
	for (int i = 0; i < sz; i++) weights[i] = 1.0 / sz;

	/* Warmup */
	for (int iter = 0; iter < 20; iter++) {
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X[i * nv + sl[j]];
				local_Y[i * sz + j] = Y[i * nv + sl[j]];
			}
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X, sz, local_Y, sz, 0.0, A, sz);
		memcpy(A_orig, A, (size_t)sz * sz * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
		        work, &lwork, iwork, &info);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            sz, sz, sz, 1.0, U, sz, Vt, sz, 0.0, T_local, sz);
	}

	/* ---------- Benchmark each step ---------- */

	/* 1. Column extraction */
	double t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X[i * nv + sl[j]];
				local_Y[i * sz + j] = Y[i * nv + sl[j]];
			}
	}
	double t1 = now_sec();
	double ms_extract = (t1 - t0) / iters * 1000;

	/* 2. GEMM: A = X^T @ Y */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X, sz, local_Y, sz, 0.0, A, sz);
	}
	t1 = now_sec();
	double ms_gemm1 = (t1 - t0) / iters * 1000;

	/* 3. SVD */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		memcpy(A, A_orig, (size_t)sz * sz * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
		        work, &lwork, iwork, &info);
	}
	t1 = now_sec();
	double ms_svd = (t1 - t0) / iters * 1000;

	/* 4. GEMM: T = U @ Vt */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            sz, sz, sz, 1.0, U, sz, Vt, sz, 0.0, T_local, sz);
	}
	t1 = now_sec();
	double ms_gemm2 = (t1 - t0) / iters * 1000;

	/* 5. Scatter-add (weighted) */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		for (int i = 0; i < sz; i++) {
			int row = sl[i];
			double wi = weights[i];
			for (int j = 0; j < sz; j++) {
				T_out[row * nv + sl[j]] += T_local[i * sz + j] * wi;
			}
		}
	}
	t1 = now_sec();
	double ms_scatter = (t1 - t0) / iters * 1000;

	double ms_total = ms_extract + ms_gemm1 + ms_svd + ms_gemm2 + ms_scatter;

	printf("--- Per-step (isolated, cached) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Column extraction:", ms_extract, ms_extract * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "GEMM A=X'Y:", ms_gemm1, ms_gemm1 * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "SVD (dgesdd):", ms_svd, ms_svd * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "GEMM T=U*Vt:", ms_gemm2, ms_gemm2 * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Scatter-add:", ms_scatter, ms_scatter * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "TOTAL (sum):", ms_total, ms_total * n_sls / 1000);

	/* Allocate workspace for reflection check (dgetrf_) */
	double *det_tmp = (double *)malloc((size_t)sz * sz * sizeof(double));
	lapack_int *det_ipiv = (lapack_int *)malloc((size_t)sz * sizeof(lapack_int));

	/* 6a. Isolated: Reflection check (dgetrf_ on 121x121) */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		memcpy(det_tmp, T_local, (size_t)sz * sz * sizeof(double));
		lapack_int det_info = 0;
		dgetrf_(&lN, &lN, det_tmp, &lN, det_ipiv, &det_info);
	}
	t1 = now_sec();
	double ms_det = (t1 - t0) / iters * 1000;
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Reflection (dgetrf):", ms_det, ms_det * n_sls / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "TOTAL+det (sum):", ms_total + ms_det, (ms_total + ms_det) * n_sls / 1000);

	/* 6b. Full pipeline interleaved -- CONTIGUOUS indices */
	int **all_sls = (int **)malloc((size_t)iters * sizeof(int *));
	for (int iter = 0; iter < iters; iter++) {
		all_sls[iter] = (int *)malloc((size_t)sz * sizeof(int));
		int base = (iter * 37) % (nv - sz);
		for (int i = 0; i < sz; i++) all_sls[iter][i] = base + i;
	}

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		int *cur_sl = all_sls[iter];
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X[i * nv + cur_sl[j]];
				local_Y[i * sz + j] = Y[i * nv + cur_sl[j]];
			}
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X, sz, local_Y, sz, 0.0, A, sz);
		memcpy(A_orig, A, (size_t)sz * sz * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
		        work, &lwork, iwork, &info);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            sz, sz, sz, 1.0, U, sz, Vt, sz, 0.0, T_local, sz);
		/* Reflection check */
		memcpy(det_tmp, T_local, (size_t)sz * sz * sizeof(double));
		lapack_int det_info = 0;
		dgetrf_(&lN, &lN, det_tmp, &lN, det_ipiv, &det_info);
		/* Scatter-add */
		for (int i = 0; i < sz; i++) {
			int row = cur_sl[i];
			double wi = weights[i];
			for (int j = 0; j < sz; j++)
				T_out[row * nv + cur_sl[j]] += T_local[i * sz + j] * wi;
		}
	}
	t1 = now_sec();
	double ms_contig = (t1 - t0) / iters * 1000;

	printf("\n--- Interleaved pipeline (contiguous indices) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Full pipeline:", ms_contig, ms_contig * n_sls / 1000);

	/* 6c. Full pipeline interleaved -- SCATTERED indices (realistic) */
	for (int iter = 0; iter < iters; iter++) {
		/* Every-Nth pattern like isolated test: spread across vertex array */
		for (int i = 0; i < sz; i++)
			all_sls[iter][i] = ((iter * 37 + i * 79) % nv);
		/* Sort to be monotonic (real SLs are sorted vertex indices) */
		for (int i = 0; i < sz - 1; i++)
			for (int j = i + 1; j < sz; j++)
				if (all_sls[iter][i] > all_sls[iter][j]) {
					int tmp = all_sls[iter][i];
					all_sls[iter][i] = all_sls[iter][j];
					all_sls[iter][j] = tmp;
				}
	}

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		int *cur_sl = all_sls[iter];
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X[i * nv + cur_sl[j]];
				local_Y[i * sz + j] = Y[i * nv + cur_sl[j]];
			}
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X, sz, local_Y, sz, 0.0, A, sz);
		memcpy(A_orig, A, (size_t)sz * sz * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
		        work, &lwork, iwork, &info);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            sz, sz, sz, 1.0, U, sz, Vt, sz, 0.0, T_local, sz);
		memcpy(det_tmp, T_local, (size_t)sz * sz * sizeof(double));
		lapack_int det_info = 0;
		dgetrf_(&lN, &lN, det_tmp, &lN, det_ipiv, &det_info);
		for (int i = 0; i < sz; i++) {
			int row = cur_sl[i];
			double wi = weights[i];
			for (int j = 0; j < sz; j++)
				T_out[row * nv + cur_sl[j]] += T_local[i * sz + j] * wi;
		}
	}
	t1 = now_sec();
	double ms_scattered = (t1 - t0) / iters * 1000;

	printf("\n--- Interleaved pipeline (scattered indices) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Full pipeline:", ms_scattered, ms_scattered * n_sls / 1000);

	printf("\nActual C benchmark (1t):  53.9s  (2 hemispheres)\n");
	printf("Actual Python bench (1t): 30.2s  (2 hemispheres)\n");
	printf("Per-hemisphere:  C=~27s  Python=~15s\n");

	for (int iter = 0; iter < iters; iter++) free(all_sls[iter]);
	free(all_sls);
	free(det_tmp); free(det_ipiv);

	free(X); free(Y); free(sl);
	free(local_X); free(local_Y);
	free(A); free(A_orig); free(U); free(s); free(Vt);
	free(T_local); free(T_out); free(iwork); free(work); free(weights);
	return 0;
}
