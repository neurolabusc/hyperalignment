/*
 * ha_searchlight.c -- Searchlight weights, alignment loop
 *
 * All searchlight loops use OpenMP (when compiled with OPENMP=1) with
 * #pragma omp critical for the scatter-add into the shared sparse matrix.
 * Thread count is controlled via ha_set_num_threads() or OMP_NUM_THREADS.
 */

#include "ha_searchlight.h"
#include "ha_procrustes.h"
#include "ha_ridge.h"
#include "ha_sparse.h"
#include "ha_metal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void ha_set_num_threads(int n) {
#ifdef _OPENMP
	omp_set_num_threads(n);
#else
	(void)n;
#endif
}

int ha_get_num_threads(void) {
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

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

// Process a batch of Newton iterations + scatter-add (shared by Metal pipeline stages)
static int metal_process_batch(int32_t start, int32_t count,
                               const int32_t *sizes,
                               float **Xdata, float **Ydata, float **Adata,
                               const TSearchlights *sls_X,
                               const TSearchlights *sls_Y_actual,
                               TSparseCSC *mat, double **weights,
                               bool isReflection, bool isScaling, int32_t nt) {
	int error = kHaSuccess;
	#ifdef _OPENMP
	#pragma omp parallel for schedule(dynamic, 1)
	#endif
	for (int32_t i = 0; i < count; i++) {
		if (error != kHaSuccess) continue;
		int32_t s = start + i;
		int32_t sz = sizes[i];
		float *T_f = (float *)malloc((size_t)sz * sz * sizeof(float));
		if (!T_f) { error = kHaErrorAlloc; continue; }

		int rc = ha_polar_newton(Adata[i], T_f, sz,
		                         isReflection, isScaling,
		                         Xdata[i], nt);
		if (rc != kHaSuccess) {
			// Fall back to CPU FP32 SVD
			TMatF fX_s = { Xdata[i], nt, sz };
			TMatF fY_s = { Ydata[i], nt, sz };
			TMatF fT_s = { T_f, sz, sz };
			rc = ha_procrustes_f32(&fX_s, &fY_s, &fT_s, isReflection, isScaling);
		}

		if (rc == kHaSuccess) {
			TMat *local_T = ha_mat_alloc(sz, sz);
			if (!local_T) { free(T_f); error = kHaErrorAlloc; continue; }
			for (int32_t j = 0; j < sz * sz; j++)
				local_T->data[j] = (double)T_f[j];

			double *w = weights ? weights[s] : NULL;
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				ha_sparse_scatter_add(mat, sls_X->indices[s], sls_Y_actual->indices[s],
				                      sz, sls_Y_actual->sizes[s], local_T->data, w);
			}
			ha_mat_free(local_T);
		} else {
			error = rc;
		}
		free(T_f);
	}
	return error;
}

int ha_searchlight_procrustes(const TMat *X, const TMat *Y,
                              const TSearchlights *sls_X,
                              const TSearchlights *sls_Y,
                              TSparseCSC *mat,
                              double **weights,
                              bool isReflection, bool isScaling,
                              THaBackend backend) {
	const TSearchlights *sls_Y_actual = sls_Y ? sls_Y : sls_X;
	int32_t nt = X->rows;
	int32_t total = sls_X->count;

	// CPU FP64 path (default)
	if (backend == kHaBackendCPU64) {
		int error = kHaSuccess;
		#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic, 1)
		#endif
		for (int32_t s = 0; s < total; s++) {
			if (error != kHaSuccess) continue;
			int32_t *sl_x = sls_X->indices[s];
			int32_t *sl_y = sls_Y_actual->indices[s];
			int32_t sz_x = sls_X->sizes[s];
			int32_t sz_y = sls_Y_actual->sizes[s];

			TMat *local_X = ha_mat_alloc(nt, sz_x);
			TMat *local_Y = ha_mat_alloc(nt, sz_y);
			TMat *local_T = ha_mat_alloc(sz_x, sz_y);
			if (!local_X || !local_Y || !local_T) {
				ha_mat_free(local_X); ha_mat_free(local_Y); ha_mat_free(local_T);
				error = kHaErrorAlloc;
				continue;
			}

			ha_mat_extract_cols(X, sl_x, sz_x, local_X);
			ha_mat_extract_cols(Y, sl_y, sz_y, local_Y);

			int rc = ha_procrustes(local_X, local_Y, local_T, isReflection, isScaling);
			ha_mat_free(local_X);
			ha_mat_free(local_Y);
			if (rc != kHaSuccess) {
				ha_mat_free(local_T);
				error = rc;
				continue;
			}

			double *w = weights ? weights[s] : NULL;
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				ha_sparse_scatter_add(mat, sl_x, sl_y, sz_x, sz_y, local_T->data, w);
			}
			ha_mat_free(local_T);
		}
		return error;
	}

	// Metal path: fall back to CPU32 if Metal not available
	THaBackend actual = backend;
	if (actual == kHaBackendMetal && !ha_metal_available())
		actual = kHaBackendCPU32;

	// CPU FP32 path
	if (actual == kHaBackendCPU32) {
		int error = kHaSuccess;
		#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic, 1)
		#endif
		for (int32_t s = 0; s < total; s++) {
			if (error != kHaSuccess) continue;
			int32_t *sl_x = sls_X->indices[s];
			int32_t *sl_y = sls_Y_actual->indices[s];
			int32_t sz_x = sls_X->sizes[s];
			int32_t sz_y = sls_Y_actual->sizes[s];

			TMat *local_X = ha_mat_alloc(nt, sz_x);
			TMat *local_Y = ha_mat_alloc(nt, sz_y);
			if (!local_X || !local_Y) {
				ha_mat_free(local_X); ha_mat_free(local_Y);
				error = kHaErrorAlloc;
				continue;
			}
			ha_mat_extract_cols(X, sl_x, sz_x, local_X);
			ha_mat_extract_cols(Y, sl_y, sz_y, local_Y);

			TMatF *fX = ha_mat_to_float(local_X);
			TMatF *fY = ha_mat_to_float(local_Y);
			TMatF *fT = ha_matf_alloc(sz_x, sz_y);
			ha_mat_free(local_X);
			ha_mat_free(local_Y);
			if (!fX || !fY || !fT) {
				ha_matf_free(fX); ha_matf_free(fY); ha_matf_free(fT);
				error = kHaErrorAlloc;
				continue;
			}

			int rc = ha_procrustes_f32(fX, fY, fT, isReflection, isScaling);
			if (rc != kHaSuccess) {
				ha_matf_free(fX); ha_matf_free(fY); ha_matf_free(fT);
				error = rc;
				continue;
			}

			TMat *local_T = ha_matf_to_double(fT);
			ha_matf_free(fX); ha_matf_free(fY); ha_matf_free(fT);
			if (!local_T) { error = kHaErrorAlloc; continue; }

			double *w = weights ? weights[s] : NULL;
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				ha_sparse_scatter_add(mat, sl_x, sl_y, sz_x, sz_y, local_T->data, w);
			}
			ha_mat_free(local_T);
		}
		return error;
	}

	// ---- Metal batched + pipelined path ----
	// GPU GEMM in batches, Newton iterations parallelized with OpenMP.
	#define METAL_BATCH 256

	int32_t max_batch = METAL_BATCH < total ? METAL_BATCH : total;
	float **b_Xdata = (float **)malloc((size_t)max_batch * sizeof(float *));
	float **b_Ydata = (float **)malloc((size_t)max_batch * sizeof(float *));
	float **b_Adata = (float **)malloc((size_t)max_batch * sizeof(float *));
	int32_t *b_sizes = (int32_t *)malloc((size_t)max_batch * sizeof(int32_t));
	if (!b_Xdata || !b_Ydata || !b_Adata || !b_sizes) {
		free(b_Xdata); free(b_Ydata); free(b_Adata); free(b_sizes);
		return kHaErrorAlloc;
	}

	int result = kHaSuccess;
	HaMetalBatch *pending = NULL;
	int32_t pending_start = 0, pending_count = 0;
	float **p_Xdata = NULL, **p_Ydata = NULL, **p_Adata = NULL;
	int32_t *p_sizes = NULL;

	for (int32_t batch_start = 0; batch_start < total; batch_start += METAL_BATCH) {
		int32_t batch_end = batch_start + METAL_BATCH;
		if (batch_end > total) batch_end = total;
		int32_t batch_count = batch_end - batch_start;

		// Extract submatrices and convert to float
		for (int32_t i = 0; i < batch_count; i++) {
			int32_t s = batch_start + i;
			int32_t sz = sls_X->sizes[s];
			b_sizes[i] = sz;

			TMat *lX = ha_mat_alloc(nt, sz);
			TMat *lY = ha_mat_alloc(nt, sz);
			if (!lX || !lY) { ha_mat_free(lX); ha_mat_free(lY); result = kHaErrorAlloc; break; }
			ha_mat_extract_cols(X, sls_X->indices[s], sz, lX);
			ha_mat_extract_cols(Y, sls_Y_actual->indices[s], sls_Y_actual->sizes[s], lY);

			TMatF *fX = ha_mat_to_float(lX);
			TMatF *fY = ha_mat_to_float(lY);
			ha_mat_free(lX);
			ha_mat_free(lY);
			if (!fX || !fY) { ha_matf_free(fX); ha_matf_free(fY); result = kHaErrorAlloc; break; }

			b_Xdata[i] = fX->data; free(fX);
			b_Ydata[i] = fY->data; free(fY);
			b_Adata[i] = (float *)malloc((size_t)sz * sz * sizeof(float));
			if (!b_Adata[i]) { result = kHaErrorAlloc; break; }
		}
		if (result != kHaSuccess) break;

		// Submit this batch's GEMM to GPU (non-blocking)
		HaMetalBatch *current = ha_metal_batch_submit(batch_count, nt, b_sizes,
		                                               b_Xdata, b_Ydata);
		if (!current) { result = kHaErrorInternal; break; }

		// Process PREVIOUS batch (Newton + scatter) while GPU works on current
		if (pending) {
			int rc = ha_metal_batch_wait(pending, p_Adata);
			ha_metal_batch_free(pending);
			pending = NULL;
			if (rc != kHaSuccess) { result = rc; break; }

			rc = metal_process_batch(pending_start, pending_count, p_sizes,
			                         p_Xdata, p_Ydata, p_Adata,
			                         sls_X, sls_Y_actual, mat, weights,
			                         isReflection, isScaling, nt);
			if (rc != kHaSuccess) result = rc;

			for (int32_t i = 0; i < pending_count; i++) {
				free(p_Xdata[i]); free(p_Ydata[i]); free(p_Adata[i]);
			}
			free(p_Xdata); free(p_Ydata); free(p_Adata); free(p_sizes);
			p_Xdata = p_Ydata = p_Adata = NULL; p_sizes = NULL;
		}
		if (result != kHaSuccess) {
			ha_metal_batch_free(current);
			break;
		}

		// Current batch becomes pending
		pending = current;
		pending_start = batch_start;
		pending_count = batch_count;
		p_Xdata = (float **)malloc((size_t)batch_count * sizeof(float *));
		p_Ydata = (float **)malloc((size_t)batch_count * sizeof(float *));
		p_Adata = (float **)malloc((size_t)batch_count * sizeof(float *));
		p_sizes = (int32_t *)malloc((size_t)batch_count * sizeof(int32_t));
		if (!p_Xdata || !p_Ydata || !p_Adata || !p_sizes) {
			result = kHaErrorAlloc;
			ha_metal_batch_free(pending); pending = NULL;
			break;
		}
		memcpy(p_Xdata, b_Xdata, (size_t)batch_count * sizeof(float *));
		memcpy(p_Ydata, b_Ydata, (size_t)batch_count * sizeof(float *));
		memcpy(p_Adata, b_Adata, (size_t)batch_count * sizeof(float *));
		memcpy(p_sizes, b_sizes, (size_t)batch_count * sizeof(int32_t));
	}

	// Process final pending batch
	if (pending && result == kHaSuccess) {
		int rc = ha_metal_batch_wait(pending, p_Adata);
		ha_metal_batch_free(pending);
		pending = NULL;
		if (rc != kHaSuccess) { result = rc; goto cleanup; }

		rc = metal_process_batch(pending_start, pending_count, p_sizes,
		                         p_Xdata, p_Ydata, p_Adata,
		                         sls_X, sls_Y_actual, mat, weights,
		                         isReflection, isScaling, nt);
		if (rc != kHaSuccess) result = rc;
	}

cleanup:
	if (pending) ha_metal_batch_free(pending);
	if (p_Xdata) {
		for (int32_t i = 0; i < pending_count; i++) {
			free(p_Xdata[i]); free(p_Ydata[i]); free(p_Adata[i]);
		}
		free(p_Xdata); free(p_Ydata); free(p_Adata); free(p_sizes);
	}
	free(b_Xdata); free(b_Ydata); free(b_Adata); free(b_sizes);
	return result;
	#undef METAL_BATCH
}

int ha_searchlight_ridge(const TMat *X, const TMat *Y,
                         const TSearchlights *sls_X,
                         const TSearchlights *sls_Y,
                         TSparseCSC *mat,
                         double **weights, double alpha) {
	const TSearchlights *sls_Y_actual = sls_Y ? sls_Y : sls_X;
	int32_t nt = X->rows;
	int error = kHaSuccess;

	#ifdef _OPENMP
	#pragma omp parallel for schedule(dynamic, 1)
	#endif
	for (int32_t s = 0; s < sls_X->count; s++) {
		if (error != kHaSuccess) continue;
		int32_t *sl_x = sls_X->indices[s];
		int32_t *sl_y = sls_Y_actual->indices[s];
		int32_t sz_x = sls_X->sizes[s];
		int32_t sz_y = sls_Y_actual->sizes[s];

		TMat *local_X = ha_mat_alloc(nt, sz_x);
		TMat *local_Y = ha_mat_alloc(nt, sz_y);
		TMat *local_T = ha_mat_alloc(sz_x, sz_y);
		if (!local_X || !local_Y || !local_T) {
			ha_mat_free(local_X); ha_mat_free(local_Y); ha_mat_free(local_T);
			error = kHaErrorAlloc;
			continue;
		}

		ha_mat_extract_cols(X, sl_x, sz_x, local_X);
		ha_mat_extract_cols(Y, sl_y, sz_y, local_Y);

		int rc = ha_ridge(local_X, local_Y, alpha, local_T);
		ha_mat_free(local_X);
		ha_mat_free(local_Y);
		if (rc != kHaSuccess) {
			ha_mat_free(local_T);
			error = rc;
			continue;
		}

		double *w = weights ? weights[s] : NULL;
		#ifdef _OPENMP
		#pragma omp critical
		#endif
		{
			ha_sparse_scatter_add(mat, sl_x, sl_y, sz_x, sz_y, local_T->data, w);
		}
		ha_mat_free(local_T);
	}

	return error;
}
