/*
 * ha_searchlight.c -- Searchlight weights, alignment loop
 *
 * Python reference (searchlight.py):
 *   for sl_X, sl_Y, w in zip(sls, sls_Y, weights):
 *       t = sl_func(X[:, sl_X], Y[:, sl_Y])
 *       mat[np.ix_(sl_X, sl_Y)] += t * w[np.newaxis]
 */

#include "ha_searchlight.h"
#include "ha_procrustes.h"
#include "ha_ridge.h"
#include "ha_sparse.h"

int ha_searchlight_weights(const TSearchlights *sls, double **weights_out) {
	// Find max vertex index
	int32_t nv = 0;
	for (int32_t s = 0; s < sls->count; s++)
		for (int32_t i = 0; i < sls->sizes[s]; i++)
			if (sls->indices[s][i] >= nv)
				nv = sls->indices[s][i] + 1;

	double *weights_sum = (double *)calloc((size_t)nv, sizeof(double));
	if (!weights_sum) return kHaErrorAlloc;

	if (sls->dists == NULL) {
		// Uniform: each vertex covered by a searchlight gets +1
		for (int32_t s = 0; s < sls->count; s++)
			for (int32_t i = 0; i < sls->sizes[s]; i++)
				weights_sum[sls->indices[s][i]] += 1.0;

		for (int32_t s = 0; s < sls->count; s++) {
			int32_t sz = sls->sizes[s];
			weights_out[s] = (double *)malloc((size_t)sz * sizeof(double));
			if (!weights_out[s]) {
				for (int32_t k = 0; k < s; k++) free(weights_out[k]);
				free(weights_sum);
				return kHaErrorAlloc;
			}
			for (int32_t i = 0; i < sz; i++)
				weights_out[s][i] = 1.0 / weights_sum[sls->indices[s][i]];
		}
	} else {
		// Distance-based: w = (radius - dist) / radius
		double radius = sls->radius;
		for (int32_t s = 0; s < sls->count; s++) {
			int32_t sz = sls->sizes[s];
			for (int32_t i = 0; i < sz; i++) {
				double w = (radius - sls->dists[s][i]) / radius;
				weights_sum[sls->indices[s][i]] += w;
			}
		}

		for (int32_t s = 0; s < sls->count; s++) {
			int32_t sz = sls->sizes[s];
			weights_out[s] = (double *)malloc((size_t)sz * sizeof(double));
			if (!weights_out[s]) {
				for (int32_t k = 0; k < s; k++) free(weights_out[k]);
				free(weights_sum);
				return kHaErrorAlloc;
			}
			for (int32_t i = 0; i < sz; i++) {
				double w = (radius - sls->dists[s][i]) / radius;
				weights_out[s][i] = w / weights_sum[sls->indices[s][i]];
			}
		}
	}

	free(weights_sum);
	return kHaSuccess;
}

// Shared inner loop for searchlight alignment
static int searchlight_loop(const TMat *X, const TMat *Y,
                            const TSearchlights *sls_X,
                            const TSearchlights *sls_Y_actual,
                            TSparseCSC *mat,
                            double **weights,
                            int (*sl_func)(const TMat *, const TMat *, TMat *)) {
	int32_t nt = X->rows;

	for (int32_t s = 0; s < sls_X->count; s++) {
		int32_t *sl_x = sls_X->indices[s];
		int32_t *sl_y = sls_Y_actual->indices[s];
		int32_t sz_x = sls_X->sizes[s];
		int32_t sz_y = sls_Y_actual->sizes[s];

		// Extract local submatrices
		TMat *local_X = ha_mat_alloc(nt, sz_x);
		TMat *local_Y = ha_mat_alloc(nt, sz_y);
		TMat *local_T = ha_mat_alloc(sz_x, sz_y);
		if (!local_X || !local_Y || !local_T) {
			ha_mat_free(local_X);
			ha_mat_free(local_Y);
			ha_mat_free(local_T);
			return kHaErrorAlloc;
		}

		ha_mat_extract_cols(X, sl_x, sz_x, local_X);
		ha_mat_extract_cols(Y, sl_y, sz_y, local_Y);

		int rc = sl_func(local_X, local_Y, local_T);
		if (rc != kHaSuccess) {
			ha_mat_free(local_X);
			ha_mat_free(local_Y);
			ha_mat_free(local_T);
			return rc;
		}

		// Scatter-add into sparse matrix
		double *w = weights ? weights[s] : NULL;
		ha_sparse_scatter_add(mat, sl_x, sl_y, sz_x, sz_y, local_T->data, w);

		ha_mat_free(local_X);
		ha_mat_free(local_Y);
		ha_mat_free(local_T);
	}

	return kHaSuccess;
}

// Wrapper structs to pass parameters through the generic loop
typedef struct {
	bool isReflection;
	bool isScaling;
} TProcOpts;

typedef struct {
	double alpha;
} TRidgeOpts;

// Thread-local option storage for the callback (not ideal but simple)
static _Thread_local TProcOpts tl_proc_opts;
static _Thread_local TRidgeOpts tl_ridge_opts;

static int sl_procrustes_func(const TMat *X, const TMat *Y, TMat *T) {
	return ha_procrustes(X, Y, T, tl_proc_opts.isReflection, tl_proc_opts.isScaling);
}

static int sl_ridge_func(const TMat *X, const TMat *Y, TMat *T) {
	return ha_ridge(X, Y, tl_ridge_opts.alpha, T);
}

int ha_searchlight_procrustes(const TMat *X, const TMat *Y,
                              const TSearchlights *sls_X,
                              const TSearchlights *sls_Y,
                              TSparseCSC *mat,
                              double **weights,
                              bool isReflection, bool isScaling) {
	const TSearchlights *sls_Y_actual = sls_Y ? sls_Y : sls_X;
	tl_proc_opts.isReflection = isReflection;
	tl_proc_opts.isScaling = isScaling;
	return searchlight_loop(X, Y, sls_X, sls_Y_actual, mat, weights, sl_procrustes_func);
}

int ha_searchlight_ridge(const TMat *X, const TMat *Y,
                         const TSearchlights *sls_X,
                         const TSearchlights *sls_Y,
                         TSparseCSC *mat,
                         double **weights, double alpha) {
	const TSearchlights *sls_Y_actual = sls_Y ? sls_Y : sls_X;
	tl_ridge_opts.alpha = alpha;
	return searchlight_loop(X, Y, sls_X, sls_Y_actual, mat, weights, sl_ridge_func);
}
