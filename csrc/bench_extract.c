/*
 * bench_extract.c -- Microbenchmark for column extraction strategies.
 * Compile: cc -O2 bench_extract.c -framework Accelerate -o bench_extract
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double now_sec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
	int nt = 1818;
	int nv = 9675;
	int sz = 121;
	int iters = 5000;

	/* Source matrix */
	double *X = (double *)malloc((size_t)nt * nv * sizeof(double));
	srand(42);
	for (size_t i = 0; i < (size_t)nt * nv; i++)
		X[i] = (double)rand() / RAND_MAX - 0.5;
	double *dst = (double *)malloc((size_t)nt * sz * sizeof(double));

	/* Scattered indices (every ~80th vertex) */
	int *sl_scattered = (int *)malloc((size_t)sz * sizeof(int));
	for (int i = 0; i < sz; i++) sl_scattered[i] = i * (nv / sz);

	/* Contiguous indices */
	int *sl_contig = (int *)malloc((size_t)sz * sizeof(int));
	for (int i = 0; i < sz; i++) sl_contig[i] = 100 + i;

	/* Semi-contiguous indices (like real searchlights: ~2-3 vertex gaps) */
	int *sl_semi = (int *)malloc((size_t)sz * sizeof(int));
	{
		int base = 500;
		int idx = 0;
		for (int i = 0; i < sz; i++) {
			sl_semi[i] = base + idx;
			idx += 1 + (i % 3 == 0);  /* skip every 3rd */
		}
	}

	printf("nt=%d, nv=%d, sz=%d, iters=%d\n", nt, nv, sz, iters);
	printf("Row size: %d doubles = %.1f KB\n", nv, nv * 8.0 / 1024);
	printf("L1 cache line: 64 bytes = 8 doubles\n\n");

	/* === Method 1: Row-major (current C code) === */
	/* i.e., for each row, gather scattered columns */
	double t0, t1;

	/* Scattered */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++)
				dst[i * sz + j] = X[i * nv + sl_scattered[j]];
	t1 = now_sec();
	double ms_row_scatter = (t1 - t0) / iters * 1000;

	/* Contiguous */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++)
				dst[i * sz + j] = X[i * nv + sl_contig[j]];
	t1 = now_sec();
	double ms_row_contig = (t1 - t0) / iters * 1000;

	/* Semi-contiguous */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int i = 0; i < nt; i++)
			for (int j = 0; j < sz; j++)
				dst[i * sz + j] = X[i * nv + sl_semi[j]];
	t1 = now_sec();
	double ms_row_semi = (t1 - t0) / iters * 1000;

	printf("--- Method 1: Row-major gather (current C code) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Scattered:", ms_row_scatter, ms_row_scatter * 9675 / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Contiguous:", ms_row_contig, ms_row_contig * 9675 / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Semi-contiguous:", ms_row_semi, ms_row_semi * 9675 / 1000);

	/* === Method 2: Column-major (cblas_dcopy per column) === */
	/* Copy one column at a time using strided copy */

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int j = 0; j < sz; j++)
			cblas_dcopy(nt, &X[sl_scattered[j]], nv, &dst[j], sz);
	t1 = now_sec();
	double ms_col_scatter = (t1 - t0) / iters * 1000;

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int j = 0; j < sz; j++)
			cblas_dcopy(nt, &X[sl_contig[j]], nv, &dst[j], sz);
	t1 = now_sec();
	double ms_col_contig = (t1 - t0) / iters * 1000;

	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int j = 0; j < sz; j++)
			cblas_dcopy(nt, &X[sl_semi[j]], nv, &dst[j], sz);
	t1 = now_sec();
	double ms_col_semi = (t1 - t0) / iters * 1000;

	printf("\n--- Method 2: Column-major (cblas_dcopy per column) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Scattered:", ms_col_scatter, ms_col_scatter * 9675 / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Contiguous:", ms_col_contig, ms_col_contig * 9675 / 1000);
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Semi-contiguous:", ms_col_semi, ms_col_semi * 9675 / 1000);

	/* === Method 3: memcpy per row (contiguous only) === */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int i = 0; i < nt; i++)
			memcpy(&dst[i * sz], &X[i * nv + sl_contig[0]], (size_t)sz * sizeof(double));
	t1 = now_sec();
	double ms_memcpy_contig = (t1 - t0) / iters * 1000;

	printf("\n--- Method 3: memcpy per row (contiguous only) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Contiguous:", ms_memcpy_contig, ms_memcpy_contig * 9675 / 1000);

	/* === Method 4: Row-major with prefetch hints === */
	t0 = now_sec();
	for (int iter = 0; iter < iters; iter++)
		for (int i = 0; i < nt; i++) {
			if (i + 1 < nt)
				for (int j = 0; j < sz; j += 8)
					__builtin_prefetch(&X[(i+1) * nv + sl_scattered[j]], 0, 0);
			for (int j = 0; j < sz; j++)
				dst[i * sz + j] = X[i * nv + sl_scattered[j]];
		}
	t1 = now_sec();
	double ms_prefetch = (t1 - t0) / iters * 1000;

	printf("\n--- Method 4: Row-major with prefetch (scattered) ---\n");
	printf("%-25s %7.3f ms  (%5.1fs projected)\n", "Scattered+prefetch:", ms_prefetch, ms_prefetch * 9675 / 1000);

	free(X); free(dst);
	free(sl_scattered); free(sl_contig); free(sl_semi);
	return 0;
}
