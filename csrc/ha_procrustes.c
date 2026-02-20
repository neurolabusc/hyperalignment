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

// ---- Polar decomposition via Newton iteration (CPU, FP32) ----

static void transpose_inplace_f32(float *data, int32_t N) {
	for (int32_t i = 0; i < N; i++)
		for (int32_t j = i + 1; j < N; j++) {
			float tmp = data[i * N + j];
			data[i * N + j] = data[j * N + i];
			data[j * N + i] = tmp;
		}
}

// Returns A^{-1} in Ainv_data. Returns kHaSuccess or kHaErrorInternal if singular.
static int matrix_inverse_f32(const float *A_data, float *Ainv_data, int32_t N) {
	memcpy(Ainv_data, A_data, (size_t)N * N * sizeof(float));
	ha_lapack_int lN = N, info = 0;
	ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
	if (!ipiv) return kHaErrorAlloc;

	// Row-major A to column-major LAPACK: sees A^T.
	// sgetrf(A^T) = P * L * U,  sgetri → (A^T)^{-1}
	// Read as row-major: ((A^T)^{-1})^T = A^{-1}
	sgetrf_(&lN, &lN, Ainv_data, &lN, ipiv, &info);
	if (info != 0) { free(ipiv); return kHaErrorInternal; }

	float work_query;
	ha_lapack_int lwork = -1;
	sgetri_(&lN, Ainv_data, &lN, ipiv, &work_query, &lwork, &info);
	lwork = (ha_lapack_int)work_query;
	float *work = (float *)malloc((size_t)lwork * sizeof(float));
	if (!work) { free(ipiv); return kHaErrorAlloc; }

	sgetri_(&lN, Ainv_data, &lN, ipiv, work, &lwork, &info);
	free(work);
	free(ipiv);
	return (info == 0) ? kHaSuccess : kHaErrorInternal;
}

static float frobenius_reldiff_f32(const float *a, const float *b, int32_t n) {
	float diff_sq = 0.0f, norm_sq = 0.0f;
	for (int32_t i = 0; i < n; i++) {
		float d = a[i] - b[i];
		diff_sq += d * d;
		norm_sq += a[i] * a[i];
	}
	return sqrtf(diff_sq) / (sqrtf(norm_sq) + 1e-30f);
}

int ha_polar_newton(const float *A, float *T, int32_t N,
                    bool isReflection, bool isScaling,
                    const float *X_data, int32_t M) {
	float *Xk = (float *)malloc((size_t)N * N * sizeof(float));
	float *Xkinvt = (float *)malloc((size_t)N * N * sizeof(float));
	float *prev = (float *)malloc((size_t)N * N * sizeof(float));
	if (!Xk || !Xkinvt || !prev) {
		free(Xk); free(Xkinvt); free(prev);
		return kHaErrorAlloc;
	}

	memcpy(Xk, A, (size_t)N * N * sizeof(float));

	int32_t max_iter = 20;
	float tol = 1e-6f;
	for (int32_t iter = 0; iter < max_iter; iter++) {
		memcpy(prev, Xk, (size_t)N * N * sizeof(float));

		// matrix_inverse_f32 returns A^{-1}; transpose to get A^{-T}
		int rc = matrix_inverse_f32(Xk, Xkinvt, N);
		if (rc != kHaSuccess) {
			free(Xk); free(Xkinvt); free(prev);
			return kHaErrorInternal;
		}
		transpose_inplace_f32(Xkinvt, N);

		// X_{k+1} = (X_k + X_k^{-T}) / 2
		for (int32_t i = 0; i < N * N; i++)
			Xk[i] = 0.5f * (Xk[i] + Xkinvt[i]);

		if (frobenius_reldiff_f32(Xk, prev, N * N) < tol)
			break;
	}
	free(Xkinvt);
	free(prev);

	memcpy(T, Xk, (size_t)N * N * sizeof(float));
	free(Xk);

	// Handle reflection
	if (!isReflection) {
		float *T_tmp = (float *)malloc((size_t)N * N * sizeof(float));
		ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
		if (!T_tmp || !ipiv) { free(T_tmp); free(ipiv); return kHaErrorAlloc; }
		memcpy(T_tmp, T, (size_t)N * N * sizeof(float));

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
			for (int32_t i = 0; i < N; i++)
				T[i * N + (N - 1)] *= -1.0f;
		}
	}

	// Handle scaling
	if (isScaling && X_data) {
		// H = T^T @ A (N x N), on CPU
		float *H = (float *)malloc((size_t)N * N * sizeof(float));
		if (!H) return kHaErrorAlloc;
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		            N, N, N, 1.0f, T, N, A, N, 0.0f, H, N);

		float trace_H = 0.0f;
		for (int32_t i = 0; i < N; i++)
			trace_H += H[i * N + i];
		free(H);

		float var_sum = 0.0f;
		for (int32_t j = 0; j < N; j++) {
			float mean = 0.0f;
			for (int32_t i = 0; i < M; i++)
				mean += X_data[i * N + j];
			mean /= M;
			float var = 0.0f;
			for (int32_t i = 0; i < M; i++) {
				float diff = X_data[i * N + j] - mean;
				var += diff * diff;
			}
			var_sum += var / M;
		}

		float scale = trace_H / (var_sum * M);
		cblas_sscal(N * N, scale, T, 1);
	}

	return kHaSuccess;
}
