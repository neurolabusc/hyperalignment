/*
 * ha_ridge.c -- Ridge regression via SVD, grid search, ensemble
 *
 * Python reference (ridge.py):
 *   U, s, Vt = safe_svd(X, remove_mean=False)
 *   d = s / (alpha + s**2)
 *   betas = Vt.T @ (d[:, np.newaxis] * (U.T @ Y))
 */

#include "ha_ridge.h"
#include "ha_linalg.h"

int ha_ridge(const TMat *X, const TMat *Y, double alpha, TMat *betas) {
	int32_t M = X->rows;
	int32_t N = X->cols;
	int32_t P = Y->cols;
	int32_t K = (M < N) ? M : N;

	TMat *U; double *s; TMat *Vt;
	int rc = ha_svd_alloc(X, &U, &s, &Vt, false);
	if (rc != kHaSuccess) return rc;

	// d = s / (alpha + s^2)
	double *d = (double *)malloc((size_t)K * sizeof(double));
	if (!d) { ha_mat_free(U); free(s); ha_mat_free(Vt); return kHaErrorAlloc; }
	for (int32_t i = 0; i < K; i++)
		d[i] = s[i] / (alpha + s[i] * s[i]);

	// UT_Y = U^T @ Y, shape (K, P)
	double *UT_Y = (double *)malloc((size_t)K * P * sizeof(double));
	if (!UT_Y) {
		ha_mat_free(U); free(s); free(d); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            K, P, M, 1.0, U->data, K, Y->data, P, 0.0, UT_Y, P);

	// Scale rows of UT_Y by d: d_UT_Y[i,j] = d[i] * UT_Y[i,j]
	for (int32_t i = 0; i < K; i++)
		for (int32_t j = 0; j < P; j++)
			UT_Y[i * P + j] *= d[i];

	// betas = Vt^T @ d_UT_Y, shape (N, P)
	// Vt is (K, N), Vt^T is (N, K)
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            N, P, K, 1.0, Vt->data, N, UT_Y, P, 0.0, betas->data, P);

	ha_mat_free(U);
	free(s);
	free(d);
	free(UT_Y);
	ha_mat_free(Vt);
	return kHaSuccess;
}

int ha_ridge_vec(const TMat *X, const double *y, double alpha, double *betas_out) {
	int32_t M = X->rows;
	int32_t N = X->cols;
	int32_t K = (M < N) ? M : N;

	TMat *U; double *s; TMat *Vt;
	int rc = ha_svd_alloc(X, &U, &s, &Vt, false);
	if (rc != kHaSuccess) return rc;

	// d = s / (alpha + s^2)
	double *d = (double *)malloc((size_t)K * sizeof(double));
	if (!d) { ha_mat_free(U); free(s); ha_mat_free(Vt); return kHaErrorAlloc; }
	for (int32_t i = 0; i < K; i++)
		d[i] = s[i] / (alpha + s[i] * s[i]);

	// UT_y = U^T @ y, vector of length K
	double *UT_y = (double *)malloc((size_t)K * sizeof(double));
	if (!UT_y) {
		ha_mat_free(U); free(s); free(d); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	cblas_dgemv(CblasRowMajor, CblasTrans,
	            M, K, 1.0, U->data, K, y, 1, 0.0, UT_y, 1);

	// d_UT_y = d * UT_y element-wise
	for (int32_t i = 0; i < K; i++)
		UT_y[i] *= d[i];

	// betas = Vt^T @ d_UT_y, vector of length N
	// Vt is (K, N), Vt^T is (N, K)
	cblas_dgemv(CblasRowMajor, CblasTrans,
	            K, N, 1.0, Vt->data, N, UT_y, 1, 0.0, betas_out, 1);

	ha_mat_free(U);
	free(s);
	free(d);
	free(UT_y);
	ha_mat_free(Vt);
	return kHaSuccess;
}

int ha_ridge_grid(const TMat *X, const double *y,
                  const double *alphas, int32_t n_alphas,
                  const int32_t *npcs, int32_t n_npcs,
                  const int32_t *train_idx, int32_t train_n,
                  double *betas_out) {
	int32_t M, N;
	TMat *Xt = NULL;
	double *yt = NULL;

	// Extract training subset if specified
	if (train_idx != NULL) {
		M = train_n;
		N = X->cols;
		Xt = ha_mat_alloc(M, N);
		yt = (double *)malloc((size_t)M * sizeof(double));
		if (!Xt || !yt) { ha_mat_free(Xt); free(yt); return kHaErrorAlloc; }
		for (int32_t i = 0; i < M; i++) {
			memcpy(&Xt->data[i * N], &X->data[train_idx[i] * N], (size_t)N * sizeof(double));
			yt[i] = y[train_idx[i]];
		}
	} else {
		M = X->rows;
		N = X->cols;
		Xt = ha_mat_copy(X);
		yt = (double *)malloc((size_t)M * sizeof(double));
		if (!Xt || !yt) { ha_mat_free(Xt); free(yt); return kHaErrorAlloc; }
		memcpy(yt, y, (size_t)M * sizeof(double));
	}

	int32_t K = (M < N) ? M : N;

	// SVD of training data
	TMat *U = ha_mat_alloc(M, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);
	if (!U || !s || !Vt) {
		ha_mat_free(Xt); free(yt); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}

	int rc = ha_svd(Xt, U, s, Vt, false);
	ha_mat_free(Xt);
	if (rc != kHaSuccess) {
		free(yt); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return rc;
	}

	// d[k, a] = s[k] / (alphas[a] + s[k]^2)
	double *d = (double *)malloc((size_t)K * n_alphas * sizeof(double));
	if (!d) {
		free(yt); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	for (int32_t k = 0; k < K; k++)
		for (int32_t a = 0; a < n_alphas; a++)
			d[k * n_alphas + a] = s[k] / (alphas[a] + s[k] * s[k]);

	// UT_y = U^T @ y, vector of length K
	double *UT_y = (double *)malloc((size_t)K * sizeof(double));
	if (!UT_y) {
		free(d); free(yt); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	cblas_dgemv(CblasRowMajor, CblasTrans,
	            M, K, 1.0, U->data, K, yt, 1, 0.0, UT_y, 1);
	free(yt);

	// d_UT_y[k, a] = d[k, a] * UT_y[k]
	double *d_UT_y = (double *)malloc((size_t)K * n_alphas * sizeof(double));
	if (!d_UT_y) {
		free(d); free(UT_y); ha_mat_free(U); free(s); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}
	for (int32_t k = 0; k < K; k++)
		for (int32_t a = 0; a < n_alphas; a++)
			d_UT_y[k * n_alphas + a] = d[k * n_alphas + a] * UT_y[k];

	free(d);
	free(UT_y);

	// betas_out is (N, n_alphas, n_npcs) -- zero initialize
	size_t betas_sz = (size_t)N * n_alphas * n_npcs;
	memset(betas_out, 0, betas_sz * sizeof(double));

	// For each PC chunk, compute partial betas and accumulate
	// Python: npcs_ext = [0] + list(npcs)
	// betas[..., p] = Vt.T[:, slc] @ d_UT_y[slc]
	// Then cumsum over npc dimension.
	int32_t prev_npc = 0;
	for (int32_t p = 0; p < n_npcs; p++) {
		int32_t cur_npc = npcs[p];
		int32_t chunk_start = prev_npc;
		int32_t chunk_size = cur_npc - prev_npc;

		if (chunk_size > 0 && chunk_start < K) {
			int32_t actual_size = chunk_size;
			if (chunk_start + actual_size > K)
				actual_size = K - chunk_start;

			// partial = Vt^T[:, chunk_start:chunk_end] @ d_UT_y[chunk_start:chunk_end, :]
			// Vt is (K, N), Vt^T[:, slc] = slice of columns = Vt[slc, :]^T
			// This is (N, actual_size) @ (actual_size, n_alphas) = (N, n_alphas)
			// Stored into betas_out[:, :, p]
			for (int32_t v = 0; v < N; v++) {
				for (int32_t a = 0; a < n_alphas; a++) {
					double sum = 0.0;
					for (int32_t k = chunk_start; k < chunk_start + actual_size; k++)
						sum += Vt->data[k * N + v] * d_UT_y[k * n_alphas + a];
					betas_out[(v * n_alphas + a) * n_npcs + p] = sum;
				}
			}
		}
		prev_npc = cur_npc;
	}

	// Cumulative sum over npc dimension (last axis)
	for (int32_t v = 0; v < N; v++) {
		for (int32_t a = 0; a < n_alphas; a++) {
			for (int32_t p = 1; p < n_npcs; p++) {
				int32_t idx = (v * n_alphas + a) * n_npcs + p;
				betas_out[idx] += betas_out[idx - 1];
			}
		}
	}

	ha_mat_free(U);
	free(s);
	free(d_UT_y);
	ha_mat_free(Vt);
	return kHaSuccess;
}

int ha_ensemble_ridge(const TMat *X, const double *y,
                      const double *alphas, int32_t n_alphas,
                      const int32_t *npcs, int32_t n_npcs,
                      int32_t **train_idx_li, int32_t **test_idx_li,
                      const int32_t *train_sizes, const int32_t *test_sizes,
                      int32_t n_models,
                      double *weights_out, double *pred_out,
                      double *r2_out, double *alpha_out, int32_t *npc_out) {
	int32_t nt = X->rows;
	int32_t nv = X->cols;
	size_t grid_sz = (size_t)n_alphas * n_npcs;
	size_t betas_sz = (size_t)nv * grid_sz;

	// Accumulated test-time weights: (nt, nv, n_alphas, n_npcs)
	double *weights_test = (double *)calloc((size_t)nt * nv * grid_sz, sizeof(double));
	int32_t *count_test = (int32_t *)calloc((size_t)nt, sizeof(int32_t));
	// Averaged model weights: (nv, n_alphas, n_npcs)
	double *weights_avg = (double *)calloc(betas_sz, sizeof(double));
	double *betas = (double *)malloc(betas_sz * sizeof(double));

	if (!weights_test || !count_test || !weights_avg || !betas) {
		free(weights_test); free(count_test); free(weights_avg); free(betas);
		return kHaErrorAlloc;
	}

	for (int32_t m = 0; m < n_models; m++) {
		int rc = ha_ridge_grid(X, y, alphas, n_alphas, npcs, n_npcs,
		                       train_idx_li[m], train_sizes[m], betas);
		if (rc != kHaSuccess) {
			free(weights_test); free(count_test); free(weights_avg); free(betas);
			return rc;
		}

		// Accumulate test weights: weights_test[test_idx, :, :, :] += betas
		int32_t *test_idx = test_idx_li[m];
		int32_t test_n = test_sizes[m];
		for (int32_t t = 0; t < test_n; t++) {
			int32_t ti = test_idx[t];
			for (size_t k = 0; k < betas_sz; k++)
				weights_test[(size_t)ti * nv * grid_sz + k] += betas[k];
			count_test[ti]++;
		}

		// Accumulate model average
		for (size_t k = 0; k < betas_sz; k++)
			weights_avg[k] += betas[k];
	}
	free(betas);

	// Average
	for (size_t k = 0; k < betas_sz; k++)
		weights_avg[k] /= n_models;

	for (int32_t t = 0; t < nt; t++) {
		if (count_test[t] > 0) {
			double inv = 1.0 / count_test[t];
			for (size_t k = 0; k < (size_t)nv * grid_sz; k++)
				weights_test[(size_t)t * nv * grid_sz + k] *= inv;
		}
	}

	// Predictions: pred[t, a, p] = sum_v(X[t, v] * weights_test[t, v, a, p])
	double *pred = (double *)calloc((size_t)nt * grid_sz, sizeof(double));
	if (!pred) {
		free(weights_test); free(count_test); free(weights_avg);
		return kHaErrorAlloc;
	}

	for (int32_t t = 0; t < nt; t++) {
		for (int32_t v = 0; v < nv; v++) {
			double xval = X->data[t * nv + v];
			for (size_t g = 0; g < grid_sz; g++)
				pred[t * grid_sz + g] += xval * weights_test[(size_t)t * nv * grid_sz + (size_t)v * grid_sz + g];
		}
	}
	free(weights_test);

	// Cost = sum_t (y[t] - pred[t, a, p])^2 for each (a, p)
	double *cost = (double *)calloc(grid_sz, sizeof(double));
	double y_ss = 0.0;
	if (!cost) {
		free(count_test); free(weights_avg); free(pred);
		return kHaErrorAlloc;
	}
	for (int32_t t = 0; t < nt; t++) {
		y_ss += y[t] * y[t];
		for (size_t g = 0; g < grid_sz; g++) {
			double diff = y[t] - pred[t * grid_sz + g];
			cost[g] += diff * diff;
		}
	}

	// Find best (alpha, npc) = argmin cost
	size_t best_g = 0;
	for (size_t g = 1; g < grid_sz; g++) {
		if (cost[g] < cost[best_g])
			best_g = g;
	}
	int32_t best_a = (int32_t)(best_g / n_npcs);
	int32_t best_p = (int32_t)(best_g % n_npcs);

	*alpha_out = alphas[best_a];
	*npc_out = npcs[best_p];
	*r2_out = 1.0 - cost[best_g] / y_ss;

	// Extract best weights and predictions
	for (int32_t v = 0; v < nv; v++)
		weights_out[v] = weights_avg[v * grid_sz + best_g];
	for (int32_t t = 0; t < nt; t++)
		pred_out[t] = pred[t * grid_sz + best_g];

	free(cost);
	free(pred);
	free(count_test);
	free(weights_avg);
	return kHaSuccess;
}
