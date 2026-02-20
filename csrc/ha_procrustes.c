/*
 * ha_procrustes.c -- Orthogonal Procrustes algorithm
 *
 * Minimizes ||X @ T - Y||_F via SVD of the cross-correlation matrix.
 *
 * Python reference (procrustes.py):
 *   A = Y.T.dot(X).T    # equivalent to X.T @ Y
 *   U, s, Vt = safe_svd(A, remove_mean=False)
 *   T = U @ Vt
 *   if not reflection: check det(T), flip if negative
 *   if scaling: scale = sum(s) / (var(X) * M)
 */

#include "ha_procrustes.h"
#include "ha_linalg.h"

int ha_procrustes(const TMat *X, const TMat *Y, TMat *T,
                  bool isReflection, bool isScaling) {
	int32_t M = X->rows;
	int32_t N = X->cols;

	// A = X^T @ Y, shape (N, N)
	TMat *A = ha_mat_alloc(N, N);
	if (!A) return kHaErrorAlloc;

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            N, N, M, 1.0, X->data, N, Y->data, N, 0.0, A->data, N);

	// SVD of A: A = U * diag(s) * Vt
	int32_t K = N;  // A is square N x N
	TMat *U = ha_mat_alloc(N, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);
	if (!U || !s || !Vt) {
		ha_mat_free(A);
		ha_mat_free(U);
		free(s);
		ha_mat_free(Vt);
		return kHaErrorAlloc;
	}

	int rc = ha_svd(A, U, s, Vt, false);
	if (rc != kHaSuccess) {
		ha_mat_free(A);
		ha_mat_free(U);
		free(s);
		ha_mat_free(Vt);
		return rc;
	}

	// T = U @ Vt, shape (N, N)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            N, N, K, 1.0, U->data, K, Vt->data, N, 0.0, T->data, N);

	// Handle reflection constraint
	if (!isReflection) {
		// Compute det(T) via LU factorization
		// det(T) = det(T^T) so passing row-major to col-major LAPACK is fine
		double *T_tmp = (double *)malloc((size_t)N * N * sizeof(double));
		ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
		if (!T_tmp || !ipiv) {
			free(T_tmp);
			free(ipiv);
			ha_mat_free(A);
			ha_mat_free(U);
			free(s);
			ha_mat_free(Vt);
			return kHaErrorAlloc;
		}
		memcpy(T_tmp, T->data, (size_t)N * N * sizeof(double));

		ha_lapack_int lN = N, info = 0;
		dgetrf_(&lN, &lN, T_tmp, &lN, ipiv, &info);

		double sign = 1.0;
		for (int32_t i = 0; i < N; i++) {
			if (T_tmp[i * N + i] < 0.0) sign = -sign;
			if (ipiv[i] != i + 1) sign = -sign;
		}

		free(T_tmp);
		free(ipiv);

		if (sign < 0.0) {
			// Flip last singular value and correct T
			s[K - 1] *= -1.0;
			// T -= 2 * outer(U[:, K-1], Vt[K-1, :])
			// U[:, K-1]: column K-1 of U (row-major), stride = U->cols
			// Vt[K-1, :]: row K-1 of Vt (row-major), stride = 1
			cblas_dger(CblasRowMajor, N, N, -2.0,
			           &U->data[K - 1], U->cols,
			           &Vt->data[(K - 1) * Vt->cols], 1,
			           T->data, T->cols);
		}
	}

	// Handle scaling
	if (isScaling) {
		double s_sum = 0.0;
		for (int32_t i = 0; i < K; i++)
			s_sum += s[i];

		// var(X, axis=0).sum() * M  -- population variance per column, summed, times M
		double var_sum = 0.0;
		for (int32_t j = 0; j < N; j++) {
			double mean = 0.0;
			for (int32_t i = 0; i < M; i++)
				mean += X->data[i * N + j];
			mean /= M;
			double var = 0.0;
			for (int32_t i = 0; i < M; i++) {
				double diff = X->data[i * N + j] - mean;
				var += diff * diff;
			}
			var_sum += var / M;
		}

		double scale = s_sum / (var_sum * M);
		cblas_dscal(N * N, scale, T->data, 1);
	}

	ha_mat_free(A);
	ha_mat_free(U);
	free(s);
	ha_mat_free(Vt);
	return kHaSuccess;
}

int ha_procrustes_f32(const TMatF *X, const TMatF *Y, TMatF *T,
                      bool isReflection, bool isScaling) {
	int32_t M = X->rows;
	int32_t N = X->cols;

	// A = X^T @ Y, shape (N, N)
	TMatF *A = ha_matf_alloc(N, N);
	if (!A) return kHaErrorAlloc;

	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            N, N, M, 1.0f, X->data, N, Y->data, N, 0.0f, A->data, N);

	// SVD of A
	int32_t K = N;
	TMatF *U = ha_matf_alloc(N, K);
	float *s = (float *)malloc((size_t)K * sizeof(float));
	TMatF *Vt = ha_matf_alloc(K, N);
	if (!U || !s || !Vt) {
		ha_matf_free(A); ha_matf_free(U); free(s); ha_matf_free(Vt);
		return kHaErrorAlloc;
	}

	int rc = ha_svd_f32(A, U, s, Vt, false);
	if (rc != kHaSuccess) {
		ha_matf_free(A); ha_matf_free(U); free(s); ha_matf_free(Vt);
		return rc;
	}

	// T = U @ Vt
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            N, N, K, 1.0f, U->data, K, Vt->data, N, 0.0f, T->data, N);

	// Handle reflection constraint
	if (!isReflection) {
		float *T_tmp = (float *)malloc((size_t)N * N * sizeof(float));
		ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
		if (!T_tmp || !ipiv) {
			free(T_tmp); free(ipiv);
			ha_matf_free(A); ha_matf_free(U); free(s); ha_matf_free(Vt);
			return kHaErrorAlloc;
		}
		memcpy(T_tmp, T->data, (size_t)N * N * sizeof(float));

		ha_lapack_int lN = N, info = 0;
		sgetrf_(&lN, &lN, T_tmp, &lN, ipiv, &info);

		float sign = 1.0f;
		for (int32_t i = 0; i < N; i++) {
			if (T_tmp[i * N + i] < 0.0f) sign = -sign;
			if (ipiv[i] != i + 1) sign = -sign;
		}

		free(T_tmp);
		free(ipiv);

		if (sign < 0.0f) {
			s[K - 1] *= -1.0f;
			cblas_sger(CblasRowMajor, N, N, -2.0f,
			           &U->data[K - 1], U->cols,
			           &Vt->data[(K - 1) * Vt->cols], 1,
			           T->data, T->cols);
		}
	}

	// Handle scaling
	if (isScaling) {
		float s_sum = 0.0f;
		for (int32_t i = 0; i < K; i++)
			s_sum += s[i];

		float var_sum = 0.0f;
		for (int32_t j = 0; j < N; j++) {
			float mean = 0.0f;
			for (int32_t i = 0; i < M; i++)
				mean += X->data[i * N + j];
			mean /= M;
			float var = 0.0f;
			for (int32_t i = 0; i < M; i++) {
				float diff = X->data[i * N + j] - mean;
				var += diff * diff;
			}
			var_sum += var / M;
		}

		float scale = s_sum / (var_sum * M);
		cblas_sscal(N * N, scale, T->data, 1);
	}

	ha_matf_free(A);
	ha_matf_free(U);
	free(s);
	ha_matf_free(Vt);
	return kHaSuccess;
}
