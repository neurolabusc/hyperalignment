/*
 * bench_colmajor.c -- Compare row-major vs column-major extraction + GEMM.
 * Compile: cc -O2 bench_colmajor.c -framework Accelerate -o bench_colmajor
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
	int nt = 1818;
	int nv = 9675;
	int sz = 121;
	int iters = 2000;
	int n_sls = 9675;

	printf("nt=%d, nv=%d, sz=%d, n_sls=%d\n\n", nt, nv, sz, n_sls);

	/* Allocate source matrices in BOTH layouts */
	double *X_row = (double *)malloc((size_t)nt * nv * sizeof(double));  /* row-major: X[i,j] = X_row[i*nv+j] */
	double *X_col = (double *)malloc((size_t)nt * nv * sizeof(double));  /* col-major: X[i,j] = X_col[j*nt+i] */
	double *Y_row = (double *)malloc((size_t)nt * nv * sizeof(double));
	double *Y_col = (double *)malloc((size_t)nt * nv * sizeof(double));

	srand(42);
	for (int i = 0; i < nt; i++)
		for (int j = 0; j < nv; j++) {
			double vx = (double)rand() / RAND_MAX - 0.5;
			double vy = (double)rand() / RAND_MAX - 0.5;
			X_row[i * nv + j] = vx;
			X_col[j * nt + i] = vx;
			Y_row[i * nv + j] = vy;
			Y_col[j * nt + i] = vy;
		}

	/* Scattered searchlight indices (realistic: spanning full array, ~80 apart) */
	int *sl = (int *)malloc((size_t)sz * sizeof(int));
	for (int i = 0; i < sz; i++) sl[i] = i * (nv / sz);

	/* Workspaces */
	double *local_X = (double *)malloc((size_t)nt * sz * sizeof(double));
	double *local_Y = (double *)malloc((size_t)nt * sz * sizeof(double));
	double *A       = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *A_check = (double *)malloc((size_t)sz * sz * sizeof(double));
	double t0, t1;

	/* === 1. Row-major extraction + row-major GEMM (current C code) === */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X_row[i * nv + sl[j]];
				local_Y[i * sz + j] = Y_row[i * nv + sl[j]];
			}
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X, sz, local_Y, sz, 0.0, A, sz);
	}
	t1 = now_sec();
	double ms_rowmajor = (t1 - t0) / iters * 1000;
	memcpy(A_check, A, (size_t)sz * sz * sizeof(double));

	/* === 2. Column-major extraction (memcpy columns) + col-major GEMM === */
	/* local_X is col-major: local_X[i,j] = local_X[j*nt+i], column j is contiguous */
	double *local_X_cm = (double *)malloc((size_t)nt * sz * sizeof(double));
	double *local_Y_cm = (double *)malloc((size_t)nt * sz * sizeof(double));

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		for (int j = 0; j < sz; j++) {
			memcpy(&local_X_cm[j * nt], &X_col[sl[j] * nt], (size_t)nt * sizeof(double));
			memcpy(&local_Y_cm[j * nt], &Y_col[sl[j] * nt], (size_t)nt * sizeof(double));
		}
		/* A = X^T @ Y  with column-major local matrices */
		/* X is col-major (nt × sz), ld=nt; A is col-major (sz × sz), ld=sz */
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X_cm, nt, local_Y_cm, nt, 0.0, A, sz);
	}
	t1 = now_sec();
	double ms_colmajor = (t1 - t0) / iters * 1000;

	/* Verify results match */
	double max_diff = 0;
	for (int i = 0; i < sz * sz; i++) {
		double d = A[i] - A_check[i];
		if (d < 0) d = -d;
		if (d > max_diff) max_diff = d;
	}

	printf("--- Extract + GEMM (scattered indices, both X and Y) ---\n");
	printf("%-35s %7.3f ms  (%5.1fs projected)\n", "Row-major (current):", ms_rowmajor, ms_rowmajor * n_sls / 1000);
	printf("%-35s %7.3f ms  (%5.1fs projected)\n", "Column-major (memcpy columns):", ms_colmajor, ms_colmajor * n_sls / 1000);
	printf("Speedup: %.2fx\n", ms_rowmajor / ms_colmajor);
	printf("Max A diff: %e (should be ~0)\n", max_diff);

	/* === 3. Full pipeline comparison === */

	/* SVD workspace */
	double *U  = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *s  = (double *)malloc((size_t)sz * sizeof(double));
	double *Vt = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *A_orig = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *T_local = (double *)malloc((size_t)sz * sz * sizeof(double));
	double *det_tmp = (double *)malloc((size_t)sz * sz * sizeof(double));
	lapack_int *iwork = (lapack_int *)malloc(8 * (size_t)sz * sizeof(lapack_int));
	lapack_int *det_ipiv = (lapack_int *)malloc((size_t)sz * sizeof(lapack_int));
	double *T_out = (double *)calloc((size_t)nv * nv, sizeof(double));
	double *weights = (double *)malloc((size_t)sz * sizeof(double));
	for (int i = 0; i < sz; i++) weights[i] = 1.0 / sz;

	lapack_int lN = sz, info = 0, lwork = -1;
	double work_query;
	char jobz = 'S';
	dgesdd_(&jobz, &lN, &lN, A, &lN, s, Vt, &lN, U, &lN,
	        &work_query, &lwork, iwork, &info);
	lwork = (lapack_int)work_query;
	double *work = (double *)malloc((size_t)lwork * sizeof(double));

	/* Create varying searchlight indices */
	int **all_sls = (int **)malloc((size_t)iters * sizeof(int *));
	for (int iter = 0; iter < iters; iter++) {
		all_sls[iter] = (int *)malloc((size_t)sz * sizeof(int));
		for (int i = 0; i < sz; i++)
			all_sls[iter][i] = ((iter * 37 + i * 79) % nv);
		/* Sort */
		for (int i = 0; i < sz - 1; i++)
			for (int j = i + 1; j < sz; j++)
				if (all_sls[iter][i] > all_sls[iter][j]) {
					int tmp = all_sls[iter][i];
					all_sls[iter][i] = all_sls[iter][j];
					all_sls[iter][j] = tmp;
				}
	}

	/* 3a. Full pipeline: row-major */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		int *cur_sl = all_sls[iter];
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++) {
				local_X[i * sz + j] = X_row[i * nv + cur_sl[j]];
				local_Y[i * sz + j] = Y_row[i * nv + cur_sl[j]];
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
	double ms_full_row = (t1 - t0) / iters * 1000;

	/* 3b. Full pipeline: column-major extraction, rest stays row-major */
	memset(T_out, 0, (size_t)nv * nv * sizeof(double));
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++) {
		int *cur_sl = all_sls[iter];
		/* Column-major extraction */
		for (int j = 0; j < sz; j++) {
			memcpy(&local_X_cm[j * nt], &X_col[cur_sl[j] * nt], (size_t)nt * sizeof(double));
			memcpy(&local_Y_cm[j * nt], &Y_col[cur_sl[j] * nt], (size_t)nt * sizeof(double));
		}
		/* GEMM with col-major input, row-major A output */
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		            sz, sz, nt, 1.0, local_X_cm, nt, local_Y_cm, nt, 0.0, A, sz);
		/* A is now in col-major (sz × sz). For SVD, LAPACK expects col-major, which this is! */
		/* But the rest of our code (SVD trick, scatter-add) expects row-major A. */
		/* For square matrix, col-major A = row-major A^T. We need SVD(A). */
		/* dgesdd with col-major A gives SVD of A directly. */
		/* Current code: row-major A passed to LAPACK (expects col-major) = SVD(A^T). */
		/* The row-major trick: dgesdd sees A^T, returns U,s,Vt of A^T, then swap U<->Vt. */
		/* With col-major A: dgesdd sees A directly, returns true U,s,Vt. */
		/* So we DON'T swap U/Vt outputs here: */
		memcpy(A_orig, A, (size_t)sz * sz * sizeof(double));
		info = 0;
		dgesdd_(&jobz, &lN, &lN, A, &lN, s, U, &lN, Vt, &lN,
		        work, &lwork, iwork, &info);
		/* T = U @ Vt (col-major) */
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		            sz, sz, sz, 1.0, U, sz, Vt, sz, 0.0, T_local, sz);
		/* T_local is col-major. For scatter-add, we need T_local[i,j] = T_local[j*sz+i]. */
		/* Reflection check: det(T) via LU. T_local is col-major, LAPACK expects col-major: */
		memcpy(det_tmp, T_local, (size_t)sz * sz * sizeof(double));
		lapack_int det_info = 0;
		dgetrf_(&lN, &lN, det_tmp, &lN, det_ipiv, &det_info);
		/* Scatter-add: T_local is col-major, so T_local[i,j] = T_local[j*sz+i] */
		for (int i = 0; i < sz; i++) {
			int row = cur_sl[i];
			double wi = weights[i];
			for (int j = 0; j < sz; j++)
				T_out[row * nv + cur_sl[j]] += T_local[j * sz + i] * wi;  /* col-major access */
		}
	}
	t1 = now_sec();
	double ms_full_col = (t1 - t0) / iters * 1000;

	printf("\n--- Full pipeline (scattered indices, with SVD + reflection check) ---\n");
	printf("%-35s %7.3f ms  (%5.1fs projected)\n", "Row-major (current C code):", ms_full_row, ms_full_row * n_sls / 1000);
	printf("%-35s %7.3f ms  (%5.1fs projected)\n", "Column-major extraction:", ms_full_col, ms_full_col * n_sls / 1000);
	printf("Speedup: %.2fx\n", ms_full_row / ms_full_col);

	printf("\nPython reference: ~1.744 ms/SL (16.9s projected)\n");

	for (int iter = 0; iter < iters; iter++) free(all_sls[iter]);
	free(all_sls);
	free(X_row); free(X_col); free(Y_row); free(Y_col);
	free(sl); free(local_X); free(local_Y); free(local_X_cm); free(local_Y_cm);
	free(A); free(A_check); free(A_orig);
	free(U); free(s); free(Vt); free(T_local);
	free(det_tmp); free(det_ipiv);
	free(T_out); free(weights); free(iwork); free(work);
	return 0;
}
