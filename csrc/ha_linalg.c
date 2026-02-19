/*
 * ha_linalg.c -- SVD, PCA, column mean removal, column z-score
 *
 * Row-major SVD trick:
 *   A row-major matrix X(M,N) stored as X[i*N+j] is identical in memory
 *   to a column-major matrix X^T(N,M) stored as X^T[j*M+i] with M=leading dim N.
 *
 *   LAPACK dgesdd expects column-major. We pass our row-major X(M,N) as if
 *   it were column-major X^T(N,M), calling dgesdd("S", N, M, ...).
 *
 *   LAPACK decomposes X^T = U_lap * diag(s) * Vt_lap (column-major).
 *   Transposing: X = Vt_lap^T * diag(s) * U_lap^T.
 *
 *   LAPACK's U_lap is (N,K) column-major = (K,N) row-major -> this is our Vt.
 *   LAPACK's Vt_lap is (K,M) column-major = (M,K) row-major -> this is our U.
 *
 *   So we swap the output buffer assignments: pass our Vt buffer as LAPACK's U,
 *   and our U buffer as LAPACK's Vt. No explicit transpose needed.
 */

#include "ha_linalg.h"

int ha_svd(TMat *X, TMat *U, double *s, TMat *Vt, bool isRemoveMean) {
	if (isRemoveMean)
		ha_remove_col_mean(X);

	int32_t M = X->rows;
	int32_t N = X->cols;
	int32_t K = (M < N) ? M : N;

	// Back up X for fallback (dgesdd destroys the input)
	size_t data_sz = (size_t)M * N * sizeof(double);
	double *X_backup = (double *)malloc(data_sz);
	if (!X_backup) return kHaErrorAlloc;
	memcpy(X_backup, X->data, data_sz);

	ha_lapack_int lM = M, lN = N;
	ha_lapack_int lwork = -1;
	ha_lapack_int info = 0;
	double work_query;
	ha_lapack_int *iwork = (ha_lapack_int *)malloc(8 * K * sizeof(ha_lapack_int));
	if (!iwork) { free(X_backup); return kHaErrorAlloc; }

	// Row-major X(M,N) -> LAPACK sees column-major X^T(N,M)
	// LAPACK "U" (N,K col-major = K,N row-major) -> our Vt
	// LAPACK "Vt" (K,M col-major = M,K row-major) -> our U
	ha_lapack_int ldA = lN;      // leading dim of X^T in col-major = N
	ha_lapack_int ldU_lap = lN;  // leading dim of LAPACK U (N,K) col-major = N
	ha_lapack_int ldVt_lap = K;  // leading dim of LAPACK Vt (K,M) col-major = K

	// Workspace query
	char jobz = 'S';
	dgesdd_(&jobz, &lN, &lM, X->data, &ldA, s,
	        Vt->data, &ldU_lap,   // LAPACK U buffer -> our Vt
	        U->data, &ldVt_lap,   // LAPACK Vt buffer -> our U
	        &work_query, &lwork, iwork, &info);

	lwork = (ha_lapack_int)work_query;
	double *work = (double *)malloc((size_t)lwork * sizeof(double));
	if (!work) { free(iwork); free(X_backup); return kHaErrorAlloc; }

	// Actual SVD via dgesdd
	dgesdd_(&jobz, &lN, &lM, X->data, &ldA, s,
	        Vt->data, &ldU_lap,
	        U->data, &ldVt_lap,
	        work, &lwork, iwork, &info);

	if (info != 0) {
		// Fallback to dgesvd (more robust)
		memcpy(X->data, X_backup, data_sz);

		lwork = -1;
		char jobu = 'S', jobvt = 'S';
		dgesvd_(&jobu, &jobvt, &lN, &lM, X->data, &ldA, s,
		        Vt->data, &ldU_lap,
		        U->data, &ldVt_lap,
		        &work_query, &lwork, &info);

		lwork = (ha_lapack_int)work_query;
		double *work2 = (double *)realloc(work, (size_t)lwork * sizeof(double));
		if (!work2) { free(work); free(iwork); free(X_backup); return kHaErrorAlloc; }
		work = work2;

		info = 0;
		dgesvd_(&jobu, &jobvt, &lN, &lM, X->data, &ldA, s,
		        Vt->data, &ldU_lap,
		        U->data, &ldVt_lap,
		        work, &lwork, &info);
	}

	free(work);
	free(iwork);
	free(X_backup);
	return (info == 0) ? kHaSuccess : kHaErrorSvd;
}

int ha_svd_alloc(const TMat *X, TMat **U_out, double **s_out, TMat **Vt_out,
                 bool isRemoveMean) {
	int32_t M = X->rows, N = X->cols, K = (M < N) ? M : N;
	TMat *Xc = ha_mat_copy(X);
	if (!Xc) return kHaErrorAlloc;
	TMat *U = ha_mat_alloc(M, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);
	if (!U || !s || !Vt) {
		ha_mat_free(Xc); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	int rc = ha_svd(Xc, U, s, Vt, isRemoveMean);
	ha_mat_free(Xc);
	if (rc != kHaSuccess) {
		ha_mat_free(U); free(s); ha_mat_free(Vt);
		return rc;
	}
	*U_out = U; *s_out = s; *Vt_out = Vt;
	return kHaSuccess;
}

int ha_pca(const TMat *X, TMat *out, bool isRemoveMean) {
	int32_t M = X->rows;
	int32_t K = (M < X->cols) ? M : X->cols;

	TMat *U; double *s; TMat *Vt;
	int rc = ha_svd_alloc(X, &U, &s, &Vt, isRemoveMean);
	if (rc != kHaSuccess) return rc;

	// out = U * diag(s): scale each column j of U by s[j]
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < K; j++)
			out->data[i * K + j] = U->data[i * K + j] * s[j];

	ha_mat_free(U);
	free(s);
	ha_mat_free(Vt);
	return kHaSuccess;
}

void ha_remove_col_mean(TMat *X) {
	int32_t M = X->rows;
	int32_t N = X->cols;
	double *means = (double *)calloc((size_t)N, sizeof(double));
	if (!means) return;  // degrade gracefully — data left intact
	double inv_M = 1.0 / M;
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < N; j++)
			means[j] += X->data[i * N + j];
	for (int32_t j = 0; j < N; j++)
		means[j] *= inv_M;
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < N; j++)
			X->data[i * N + j] -= means[j];
	free(means);
}

void ha_zscore_columns(TMat *X) {
	int32_t M = X->rows;
	int32_t N = X->cols;
	double *sums = (double *)calloc((size_t)N, sizeof(double));
	double *sum2s = (double *)calloc((size_t)N, sizeof(double));
	if (!sums || !sum2s) { free(sums); free(sum2s); return; }
	double inv_M = 1.0 / M;
	for (int32_t i = 0; i < M; i++) {
		double *row = &X->data[i * N];
		for (int32_t j = 0; j < N; j++) {
			sums[j] += row[j];
			sum2s[j] += row[j] * row[j];
		}
	}
	// Compute mean and inv_std per column (reuse sums/sum2s arrays)
	for (int32_t j = 0; j < N; j++) {
		double mean = sums[j] * inv_M;
		double var = sum2s[j] * inv_M - mean * mean;
		sums[j] = mean;
		sum2s[j] = (var < 1e-30) ? 0.0 : 1.0 / sqrt(var);
	}
	for (int32_t i = 0; i < M; i++) {
		double *row = &X->data[i * N];
		for (int32_t j = 0; j < N; j++)
			row[j] = (row[j] - sums[j]) * sum2s[j];
	}
	free(sums);
	free(sum2s);
}
