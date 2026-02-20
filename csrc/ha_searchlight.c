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
			for (int64_t j = 0; j < (int64_t)sz * sz; j++)
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
			int32_t sz_y = sls_Y_actual->sizes[s];
			b_sizes[i] = sz;

			TMat *lX = ha_mat_alloc(nt, sz);
			TMat *lY = ha_mat_alloc(nt, sz_y);
			if (!lX || !lY) { ha_mat_free(lX); ha_mat_free(lY); result = kHaErrorAlloc; break; }
			ha_mat_extract_cols(X, sls_X->indices[s], sz, lX);
			ha_mat_extract_cols(Y, sls_Y_actual->indices[s], sz_y, lY);

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

// ---- Dense output: flat-array input, no sparse structure ----

// ---- Per-thread workspace for pre-allocated searchlight Procrustes ----

typedef struct {
	double *local_X;         // nt × max_sz
	double *local_Y;         // nt × max_sz
	double *T;               // max_sz × max_sz
	double *A;               // max_sz × max_sz (cross-correlation, destroyed by SVD)
	double *U;               // max_sz × max_sz
	double *s;               // max_sz
	double *Vt;              // max_sz × max_sz
	double *svd_backup;      // max_sz × max_sz (dgesdd fallback)
	double *svd_work;        // LAPACK workspace
	ha_lapack_int *svd_iwork; // 8 * max_sz
	ha_lapack_int svd_lwork;
	double *det_tmp;         // max_sz × max_sz (reflection check)
	ha_lapack_int *det_ipiv; // max_sz (reflection check)
} WorkspaceFP64;

static void workspace_fp64_free(WorkspaceFP64 *ws) {
	if (!ws) return;
	free(ws->local_X); free(ws->local_Y); free(ws->T);
	free(ws->A); free(ws->U); free(ws->s); free(ws->Vt);
	free(ws->svd_backup); free(ws->svd_work); free(ws->svd_iwork);
	free(ws->det_tmp); free(ws->det_ipiv);
	free(ws);
}

static WorkspaceFP64 *workspace_fp64_alloc(int32_t nt, int32_t max_sz) {
	WorkspaceFP64 *ws = (WorkspaceFP64 *)calloc(1, sizeof(WorkspaceFP64));
	if (!ws) return NULL;

	size_t nt_sz = (size_t)nt * max_sz;
	size_t sq = (size_t)max_sz * max_sz;

	ws->local_X     = (double *)malloc(nt_sz * sizeof(double));
	ws->local_Y     = (double *)malloc(nt_sz * sizeof(double));
	ws->T           = (double *)malloc(sq * sizeof(double));
	ws->A           = (double *)malloc(sq * sizeof(double));
	ws->U           = (double *)malloc(sq * sizeof(double));
	ws->s           = (double *)malloc((size_t)max_sz * sizeof(double));
	ws->Vt          = (double *)malloc(sq * sizeof(double));
	ws->svd_backup  = (double *)malloc(sq * sizeof(double));
	ws->svd_iwork   = (ha_lapack_int *)malloc(8 * (size_t)max_sz * sizeof(ha_lapack_int));
	ws->det_tmp     = (double *)malloc(sq * sizeof(double));
	ws->det_ipiv    = (ha_lapack_int *)malloc((size_t)max_sz * sizeof(ha_lapack_int));

	if (!ws->local_X || !ws->local_Y || !ws->T ||
	    !ws->A || !ws->U || !ws->s || !ws->Vt ||
	    !ws->svd_backup || !ws->svd_iwork ||
	    !ws->det_tmp || !ws->det_ipiv) {
		workspace_fp64_free(ws);
		return NULL;
	}

	// Query LAPACK for optimal dgesdd workspace at max_sz × max_sz
	ha_lapack_int lN = max_sz;
	ha_lapack_int query_lwork = -1;
	ha_lapack_int info = 0;
	double work_query;
	char jobz = 'S';
	dgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->Vt, &lN, ws->U, &lN,
	        &work_query, &query_lwork, ws->svd_iwork, &info);

	ws->svd_lwork = (ha_lapack_int)work_query;
	ws->svd_work = (double *)malloc((size_t)ws->svd_lwork * sizeof(double));
	if (!ws->svd_work) {
		workspace_fp64_free(ws);
		return NULL;
	}

	return ws;
}

// Inline Procrustes using pre-allocated workspace.
// ws->local_X (nt × sz) and ws->local_Y (nt × sz) must contain extracted column data.
// Result is written to ws->T (sz × sz).
static int procrustes_ws_fp64(WorkspaceFP64 *ws, int32_t nt, int32_t sz,
                              bool isReflection, bool isScaling) {
	// A = X^T @ Y
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            sz, sz, nt, 1.0, ws->local_X, sz, ws->local_Y, sz, 0.0, ws->A, sz);

	// Backup A for dgesdd fallback
	size_t a_bytes = (size_t)sz * sz * sizeof(double);
	memcpy(ws->svd_backup, ws->A, a_bytes);

	// SVD of A (row-major trick: N×N square, swap U/Vt outputs)
	ha_lapack_int lN = sz;
	ha_lapack_int info = 0;
	ha_lapack_int lwork = ws->svd_lwork;
	char jobz = 'S';

	dgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->Vt, &lN, ws->U, &lN,
	        ws->svd_work, &lwork, ws->svd_iwork, &info);

	if (info != 0) {
		// Fallback to dgesvd (more robust, uses same workspace)
		memcpy(ws->A, ws->svd_backup, a_bytes);
		char jobu = 'S', jobvt = 'S';
		info = 0;
		dgesvd_(&jobu, &jobvt, &lN, &lN, ws->A, &lN, ws->s,
		        ws->Vt, &lN, ws->U, &lN,
		        ws->svd_work, &lwork, &info);
		if (info != 0) return kHaErrorSvd;
	}

	// T = U @ Vt
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            sz, sz, sz, 1.0, ws->U, sz, ws->Vt, sz, 0.0, ws->T, sz);

	// Reflection check
	if (!isReflection) {
		memcpy(ws->det_tmp, ws->T, a_bytes);
		ha_lapack_int det_info = 0;
		dgetrf_(&lN, &lN, ws->det_tmp, &lN, ws->det_ipiv, &det_info);

		double sign = 1.0;
		for (int32_t i = 0; i < sz; i++) {
			if (ws->det_tmp[i * sz + i] < 0.0) sign = -sign;
			if (ws->det_ipiv[i] != i + 1) sign = -sign;
		}
		if (sign < 0.0) {
			ws->s[sz - 1] *= -1.0;
			cblas_dger(CblasRowMajor, sz, sz, -2.0,
			           &ws->U[sz - 1], sz,
			           &ws->Vt[(sz - 1) * sz], 1,
			           ws->T, sz);
		}
	}

	// Scaling
	if (isScaling) {
		double s_sum = 0.0;
		for (int32_t i = 0; i < sz; i++) s_sum += ws->s[i];

		double var_sum = 0.0;
		for (int32_t j = 0; j < sz; j++) {
			double mean = 0.0;
			for (int32_t i = 0; i < nt; i++)
				mean += ws->local_X[i * sz + j];
			mean /= nt;
			double var = 0.0;
			for (int32_t i = 0; i < nt; i++) {
				double diff = ws->local_X[i * sz + j] - mean;
				var += diff * diff;
			}
			var_sum += var / nt;
		}

		double scale = s_sum / (var_sum * nt);
		cblas_dscal(sz * sz, scale, ws->T, 1);
	}

	return kHaSuccess;
}

// Column-major Procrustes using pre-allocated workspace.
// ws->local_X (nt × sz, col-major) and ws->local_Y (nt × sz, col-major) must be filled.
// Result is written to ws->T (sz × sz, col-major).
static int procrustes_ws_fp64_cm(WorkspaceFP64 *ws, int32_t nt, int32_t sz,
                                  bool isReflection, bool isScaling) {
	// A = X^T @ Y (col-major: local_X has ld=nt, A has ld=sz)
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
	            sz, sz, nt, 1.0, ws->local_X, nt, ws->local_Y, nt, 0.0, ws->A, sz);

	size_t a_bytes = (size_t)sz * sz * sizeof(double);
	memcpy(ws->svd_backup, ws->A, a_bytes);

	// SVD of A (col-major: LAPACK native, no swap trick)
	ha_lapack_int lN = sz;
	ha_lapack_int info = 0;
	ha_lapack_int lwork = ws->svd_lwork;
	char jobz = 'S';

	dgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->U, &lN, ws->Vt, &lN,
	        ws->svd_work, &lwork, ws->svd_iwork, &info);

	if (info != 0) {
		memcpy(ws->A, ws->svd_backup, a_bytes);
		char jobu = 'S', jobvt = 'S';
		info = 0;
		dgesvd_(&jobu, &jobvt, &lN, &lN, ws->A, &lN, ws->s,
		        ws->U, &lN, ws->Vt, &lN,
		        ws->svd_work, &lwork, &info);
		if (info != 0) return kHaErrorSvd;
	}

	// T = U @ Vt (col-major)
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	            sz, sz, sz, 1.0, ws->U, sz, ws->Vt, sz, 0.0, ws->T, sz);

	// Reflection check (LAPACK is col-major native, so dgetrf_ works directly)
	if (!isReflection) {
		memcpy(ws->det_tmp, ws->T, a_bytes);
		ha_lapack_int det_info = 0;
		dgetrf_(&lN, &lN, ws->det_tmp, &lN, ws->det_ipiv, &det_info);

		double sign = 1.0;
		for (int32_t i = 0; i < sz; i++) {
			// col-major diagonal: A[i,i] = buf[i*lda+i] = buf[i*sz+i]
			if (ws->det_tmp[i * sz + i] < 0.0) sign = -sign;
			if (ws->det_ipiv[i] != i + 1) sign = -sign;
		}
		if (sign < 0.0) {
			ws->s[sz - 1] *= -1.0;
			// Col-major: U col K-1 is &U[(K-1)*sz], stride 1
			//            Vt row K-1 is &Vt[K-1], stride sz
			cblas_dger(CblasColMajor, sz, sz, -2.0,
			           &ws->U[(sz - 1) * sz], 1,
			           &ws->Vt[sz - 1], sz,
			           ws->T, sz);
		}
	}

	// Scaling
	if (isScaling) {
		double s_sum = 0.0;
		for (int32_t i = 0; i < sz; i++) s_sum += ws->s[i];

		// Col-major local_X: column j is ws->local_X[j*nt..j*nt+nt-1]
		double var_sum = 0.0;
		for (int32_t j = 0; j < sz; j++) {
			double mean = 0.0;
			for (int32_t i = 0; i < nt; i++)
				mean += ws->local_X[j * nt + i];
			mean /= nt;
			double var = 0.0;
			for (int32_t i = 0; i < nt; i++) {
				double diff = ws->local_X[j * nt + i] - mean;
				var += diff * diff;
			}
			var_sum += var / nt;
		}

		double scale = s_sum / (var_sum * nt);
		cblas_dscal(sz * sz, scale, ws->T, 1);
	}

	return kHaSuccess;
}

// ---- FP32 workspace ----

typedef struct {
	float *fX;               // nt × max_sz (extracted + converted from double)
	float *fY;               // nt × max_sz
	float *fT;               // max_sz × max_sz
	float *A;                // max_sz × max_sz
	float *U;                // max_sz × max_sz
	float *s;                // max_sz
	float *Vt;               // max_sz × max_sz
	float *svd_backup;       // max_sz × max_sz
	float *svd_work;         // LAPACK workspace
	ha_lapack_int *svd_iwork; // 8 * max_sz
	ha_lapack_int svd_lwork;
	float *det_tmp;          // max_sz × max_sz
	ha_lapack_int *det_ipiv; // max_sz
} WorkspaceFP32;

static void workspace_fp32_free(WorkspaceFP32 *ws) {
	if (!ws) return;
	free(ws->fX); free(ws->fY); free(ws->fT);
	free(ws->A); free(ws->U); free(ws->s); free(ws->Vt);
	free(ws->svd_backup); free(ws->svd_work); free(ws->svd_iwork);
	free(ws->det_tmp); free(ws->det_ipiv);
	free(ws);
}

static WorkspaceFP32 *workspace_fp32_alloc(int32_t nt, int32_t max_sz) {
	WorkspaceFP32 *ws = (WorkspaceFP32 *)calloc(1, sizeof(WorkspaceFP32));
	if (!ws) return NULL;

	size_t nt_sz = (size_t)nt * max_sz;
	size_t sq = (size_t)max_sz * max_sz;

	ws->fX          = (float *)malloc(nt_sz * sizeof(float));
	ws->fY          = (float *)malloc(nt_sz * sizeof(float));
	ws->fT          = (float *)malloc(sq * sizeof(float));
	ws->A           = (float *)malloc(sq * sizeof(float));
	ws->U           = (float *)malloc(sq * sizeof(float));
	ws->s           = (float *)malloc((size_t)max_sz * sizeof(float));
	ws->Vt          = (float *)malloc(sq * sizeof(float));
	ws->svd_backup  = (float *)malloc(sq * sizeof(float));
	ws->svd_iwork   = (ha_lapack_int *)malloc(8 * (size_t)max_sz * sizeof(ha_lapack_int));
	ws->det_tmp     = (float *)malloc(sq * sizeof(float));
	ws->det_ipiv    = (ha_lapack_int *)malloc((size_t)max_sz * sizeof(ha_lapack_int));

	if (!ws->fX || !ws->fY || !ws->fT ||
	    !ws->A || !ws->U || !ws->s || !ws->Vt ||
	    !ws->svd_backup || !ws->svd_iwork ||
	    !ws->det_tmp || !ws->det_ipiv) {
		workspace_fp32_free(ws);
		return NULL;
	}

	ha_lapack_int lN = max_sz;
	ha_lapack_int query_lwork = -1;
	ha_lapack_int info = 0;
	float work_query;
	char jobz = 'S';
	sgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->Vt, &lN, ws->U, &lN,
	        &work_query, &query_lwork, ws->svd_iwork, &info);

	ws->svd_lwork = (ha_lapack_int)work_query;
	ws->svd_work = (float *)malloc((size_t)ws->svd_lwork * sizeof(float));
	if (!ws->svd_work) {
		workspace_fp32_free(ws);
		return NULL;
	}

	return ws;
}

// Extract columns from double source directly into float destination
static inline void extract_cols_d2f(const double *src, int32_t src_cols,
                                     const int32_t *col_idx, int32_t n_cols,
                                     float *dst, int32_t rows) {
	for (int32_t i = 0; i < rows; i++)
		for (int32_t j = 0; j < n_cols; j++)
			dst[i * n_cols + j] = (float)src[i * src_cols + col_idx[j]];
}

// Inline FP32 Procrustes using pre-allocated workspace.
// ws->fX (nt × sz) and ws->fY (nt × sz) must contain extracted+converted data.
// Result is written to ws->fT (sz × sz).
static int procrustes_ws_fp32(WorkspaceFP32 *ws, int32_t nt, int32_t sz,
                              bool isReflection, bool isScaling) {
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            sz, sz, nt, 1.0f, ws->fX, sz, ws->fY, sz, 0.0f, ws->A, sz);

	size_t a_bytes = (size_t)sz * sz * sizeof(float);
	memcpy(ws->svd_backup, ws->A, a_bytes);

	ha_lapack_int lN = sz;
	ha_lapack_int info = 0;
	ha_lapack_int lwork = ws->svd_lwork;
	char jobz = 'S';

	sgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->Vt, &lN, ws->U, &lN,
	        ws->svd_work, &lwork, ws->svd_iwork, &info);

	if (info != 0) {
		memcpy(ws->A, ws->svd_backup, a_bytes);
		char jobu = 'S', jobvt = 'S';
		info = 0;
		sgesvd_(&jobu, &jobvt, &lN, &lN, ws->A, &lN, ws->s,
		        ws->Vt, &lN, ws->U, &lN,
		        ws->svd_work, &lwork, &info);
		if (info != 0) return kHaErrorSvd;
	}

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            sz, sz, sz, 1.0f, ws->U, sz, ws->Vt, sz, 0.0f, ws->fT, sz);

	if (!isReflection) {
		memcpy(ws->det_tmp, ws->fT, a_bytes);
		ha_lapack_int det_info = 0;
		sgetrf_(&lN, &lN, ws->det_tmp, &lN, ws->det_ipiv, &det_info);

		float sign = 1.0f;
		for (int32_t i = 0; i < sz; i++) {
			if (ws->det_tmp[i * sz + i] < 0.0f) sign = -sign;
			if (ws->det_ipiv[i] != i + 1) sign = -sign;
		}
		if (sign < 0.0f) {
			ws->s[sz - 1] *= -1.0f;
			cblas_sger(CblasRowMajor, sz, sz, -2.0f,
			           &ws->U[sz - 1], sz,
			           &ws->Vt[(sz - 1) * sz], 1,
			           ws->fT, sz);
		}
	}

	if (isScaling) {
		float s_sum = 0.0f;
		for (int32_t i = 0; i < sz; i++) s_sum += ws->s[i];

		float var_sum = 0.0f;
		for (int32_t j = 0; j < sz; j++) {
			float mean = 0.0f;
			for (int32_t i = 0; i < nt; i++)
				mean += ws->fX[i * sz + j];
			mean /= nt;
			float var = 0.0f;
			for (int32_t i = 0; i < nt; i++) {
				float diff = ws->fX[i * sz + j] - mean;
				var += diff * diff;
			}
			var_sum += var / nt;
		}

		float scale = s_sum / (var_sum * nt);
		cblas_sscal(sz * sz, scale, ws->fT, 1);
	}

	return kHaSuccess;
}

// Column-major FP32 Procrustes using pre-allocated workspace.
// ws->fX (nt × sz, col-major) and ws->fY (nt × sz, col-major) must be filled.
// Result is written to ws->fT (sz × sz, col-major).
static int procrustes_ws_fp32_cm(WorkspaceFP32 *ws, int32_t nt, int32_t sz,
                                  bool isReflection, bool isScaling) {
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
	            sz, sz, nt, 1.0f, ws->fX, nt, ws->fY, nt, 0.0f, ws->A, sz);

	size_t a_bytes = (size_t)sz * sz * sizeof(float);
	memcpy(ws->svd_backup, ws->A, a_bytes);

	ha_lapack_int lN = sz;
	ha_lapack_int info = 0;
	ha_lapack_int lwork = ws->svd_lwork;
	char jobz = 'S';

	sgesdd_(&jobz, &lN, &lN, ws->A, &lN, ws->s,
	        ws->U, &lN, ws->Vt, &lN,
	        ws->svd_work, &lwork, ws->svd_iwork, &info);

	if (info != 0) {
		memcpy(ws->A, ws->svd_backup, a_bytes);
		char jobu = 'S', jobvt = 'S';
		info = 0;
		sgesvd_(&jobu, &jobvt, &lN, &lN, ws->A, &lN, ws->s,
		        ws->U, &lN, ws->Vt, &lN,
		        ws->svd_work, &lwork, &info);
		if (info != 0) return kHaErrorSvd;
	}

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	            sz, sz, sz, 1.0f, ws->U, sz, ws->Vt, sz, 0.0f, ws->fT, sz);

	if (!isReflection) {
		memcpy(ws->det_tmp, ws->fT, a_bytes);
		ha_lapack_int det_info = 0;
		sgetrf_(&lN, &lN, ws->det_tmp, &lN, ws->det_ipiv, &det_info);

		float sign = 1.0f;
		for (int32_t i = 0; i < sz; i++) {
			if (ws->det_tmp[i * sz + i] < 0.0f) sign = -sign;
			if (ws->det_ipiv[i] != i + 1) sign = -sign;
		}
		if (sign < 0.0f) {
			ws->s[sz - 1] *= -1.0f;
			cblas_sger(CblasColMajor, sz, sz, -2.0f,
			           &ws->U[(sz - 1) * sz], 1,
			           &ws->Vt[sz - 1], sz,
			           ws->fT, sz);
		}
	}

	if (isScaling) {
		float s_sum = 0.0f;
		for (int32_t i = 0; i < sz; i++) s_sum += ws->s[i];

		float var_sum = 0.0f;
		for (int32_t j = 0; j < sz; j++) {
			float mean = 0.0f;
			for (int32_t i = 0; i < nt; i++)
				mean += ws->fX[j * nt + i];
			mean /= nt;
			float var = 0.0f;
			for (int32_t i = 0; i < nt; i++) {
				float diff = ws->fX[j * nt + i] - mean;
				var += diff * diff;
			}
			var_sum += var / nt;
		}

		float scale = s_sum / (var_sum * nt);
		cblas_sscal(sz * sz, scale, ws->fT, 1);
	}

	return kHaSuccess;
}

// Process a batch of Newton iterations + dense scatter-add (Metal pipeline)
static int metal_process_batch_dense(int32_t start, int32_t count,
                                      const int32_t *sizes,
                                      float **Xdata, float **Ydata, float **Adata,
                                      const int32_t *sl_indices,
                                      const int32_t *sl_offsets,
                                      const double *weights,
                                      double *T_out, int32_t nv,
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
			TMatF fX_s = { Xdata[i], nt, sz };
			TMatF fY_s = { Ydata[i], nt, sz };
			TMatF fT_s = { T_f, sz, sz };
			rc = ha_procrustes_f32(&fX_s, &fY_s, &fT_s, isReflection, isScaling);
		}

		if (rc == kHaSuccess) {
			int32_t off = sl_offsets[s];
			const int32_t *sl = &sl_indices[off];
			const double *w = &weights[off];
			for (int32_t ii = 0; ii < sz; ii++) {
				int32_t row = sl[ii];
				double wi = w[ii];
				for (int32_t jj = 0; jj < sz; jj++) {
					double val = (double)T_f[ii * sz + jj] * wi;
					#ifdef _OPENMP
					#pragma omp atomic
					#endif
					T_out[row * nv + sl[jj]] += val;
				}
			}
		} else {
			error = rc;
		}
		free(T_f);
	}
	return error;
}

int ha_searchlight_procrustes_dense(
    const double *X_data, const double *Y_data,
    int32_t nt, int32_t nv,
    const int32_t *sl_indices,
    const int32_t *sl_offsets,
    const double *sl_dists,
    int32_t count,
    double radius,
    double *T_out,
    bool isReflection, bool isScaling,
    THaBackend backend,
    bool col_major)
{
	if (!X_data || !Y_data || !sl_indices || !sl_offsets || !T_out)
		return kHaErrorArg;
	if (nt <= 0 || nv <= 0 || count <= 0)
		return kHaErrorArg;

	int32_t total_elems = sl_offsets[count];

	// ---- Compute flat weights array ----
	double *vert_wsum = (double *)calloc((size_t)nv, sizeof(double));
	double *weights = (double *)malloc((size_t)total_elems * sizeof(double));
	if (!vert_wsum || !weights) {
		free(vert_wsum); free(weights);
		return kHaErrorAlloc;
	}

	if (sl_dists == NULL) {
		for (int32_t k = 0; k < total_elems; k++)
			vert_wsum[sl_indices[k]] += 1.0;
		for (int32_t s = 0; s < count; s++) {
			int32_t off = sl_offsets[s];
			int32_t sz = sl_offsets[s + 1] - off;
			for (int32_t i = 0; i < sz; i++)
				weights[off + i] = 1.0 / vert_wsum[sl_indices[off + i]];
		}
	} else {
		for (int32_t k = 0; k < total_elems; k++) {
			double w = (radius - sl_dists[k]) / radius;
			vert_wsum[sl_indices[k]] += w;
		}
		for (int32_t s = 0; s < count; s++) {
			int32_t off = sl_offsets[s];
			int32_t sz = sl_offsets[s + 1] - off;
			for (int32_t i = 0; i < sz; i++) {
				double w = (radius - sl_dists[off + i]) / radius;
				weights[off + i] = w / vert_wsum[sl_indices[off + i]];
			}
		}
	}
	free(vert_wsum);

	// Wrap raw pointers in TMat for ha_mat_extract_cols (row-major paths only)
	// Cast away const -- extract_cols only reads from src
	TMat X_mat = { (double *)X_data, nt, nv };
	TMat Y_mat = { (double *)Y_data, nt, nv };

	// ---- CPU FP64 path ----
	if (backend == kHaBackendCPU64) {
		int32_t max_sz = 0;
		for (int32_t s = 0; s < count; s++) {
			int32_t sz = sl_offsets[s + 1] - sl_offsets[s];
			if (sz > max_sz) max_sz = sz;
		}

		int n_threads = 1;
		#ifdef _OPENMP
		n_threads = omp_get_max_threads();
		#endif

		WorkspaceFP64 **ws_arr = (WorkspaceFP64 **)calloc((size_t)n_threads, sizeof(WorkspaceFP64 *));
		if (!ws_arr) { free(weights); return kHaErrorAlloc; }
		int alloc_ok = 1;
		for (int t = 0; t < n_threads; t++) {
			ws_arr[t] = workspace_fp64_alloc(nt, max_sz);
			if (!ws_arr[t]) { alloc_ok = 0; break; }
		}
		if (!alloc_ok) {
			for (int t = 0; t < n_threads; t++) workspace_fp64_free(ws_arr[t]);
			free(ws_arr); free(weights);
			return kHaErrorAlloc;
		}

		int error = kHaSuccess;
		#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic, 1)
		#endif
		for (int32_t s = 0; s < count; s++) {
			if (error != kHaSuccess) continue;

			int tid = 0;
			#ifdef _OPENMP
			tid = omp_get_thread_num();
			#endif
			WorkspaceFP64 *ws = ws_arr[tid];

			int32_t off = sl_offsets[s];
			int32_t sz = sl_offsets[s + 1] - off;
			const int32_t *sl = &sl_indices[off];

			int rc;
			if (col_major) {
				// Col-major extraction: column j is contiguous at X_data[sl[j]*nt]
				for (int32_t j = 0; j < sz; j++) {
					memcpy(&ws->local_X[j * nt], &X_data[(size_t)sl[j] * nt],
					       (size_t)nt * sizeof(double));
					memcpy(&ws->local_Y[j * nt], &Y_data[(size_t)sl[j] * nt],
					       (size_t)nt * sizeof(double));
				}
				rc = procrustes_ws_fp64_cm(ws, nt, sz, isReflection, isScaling);
			} else {
				// Row-major extraction
				for (int32_t i = 0; i < nt; i++)
					for (int32_t j = 0; j < sz; j++) {
						ws->local_X[i * sz + j] = X_data[i * nv + sl[j]];
						ws->local_Y[i * sz + j] = Y_data[i * nv + sl[j]];
					}
				rc = procrustes_ws_fp64(ws, nt, sz, isReflection, isScaling);
			}
			if (rc != kHaSuccess) { error = rc; continue; }

			const double *w = &weights[off];
			if (col_major) {
				// T is col-major: T[i,j] = ws->T[j * sz + i]
				for (int32_t i = 0; i < sz; i++) {
					int32_t row = sl[i];
					double wi = w[i];
					for (int32_t j = 0; j < sz; j++) {
						double val = ws->T[j * sz + i] * wi;
						#ifdef _OPENMP
						#pragma omp atomic
						#endif
						T_out[row * nv + sl[j]] += val;
					}
				}
			} else {
				// T is row-major: T[i,j] = ws->T[i * sz + j]
				for (int32_t i = 0; i < sz; i++) {
					int32_t row = sl[i];
					double wi = w[i];
					for (int32_t j = 0; j < sz; j++) {
						double val = ws->T[i * sz + j] * wi;
						#ifdef _OPENMP
						#pragma omp atomic
						#endif
						T_out[row * nv + sl[j]] += val;
					}
				}
			}
		}

		for (int t = 0; t < n_threads; t++) workspace_fp64_free(ws_arr[t]);
		free(ws_arr);
		free(weights);
		return error;
	}

	// Metal path: fall back to CPU32 if Metal not available
	THaBackend actual = backend;
	if (actual == kHaBackendMetal && !ha_metal_available())
		actual = kHaBackendCPU32;

	// ---- CPU FP32 path ----
	if (actual == kHaBackendCPU32) {
		int32_t max_sz = 0;
		for (int32_t s = 0; s < count; s++) {
			int32_t sz = sl_offsets[s + 1] - sl_offsets[s];
			if (sz > max_sz) max_sz = sz;
		}

		int n_threads = 1;
		#ifdef _OPENMP
		n_threads = omp_get_max_threads();
		#endif

		WorkspaceFP32 **ws_arr = (WorkspaceFP32 **)calloc((size_t)n_threads, sizeof(WorkspaceFP32 *));
		if (!ws_arr) { free(weights); return kHaErrorAlloc; }
		int alloc_ok = 1;
		for (int t = 0; t < n_threads; t++) {
			ws_arr[t] = workspace_fp32_alloc(nt, max_sz);
			if (!ws_arr[t]) { alloc_ok = 0; break; }
		}
		if (!alloc_ok) {
			for (int t = 0; t < n_threads; t++) workspace_fp32_free(ws_arr[t]);
			free(ws_arr); free(weights);
			return kHaErrorAlloc;
		}

		int error = kHaSuccess;
		#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic, 1)
		#endif
		for (int32_t s = 0; s < count; s++) {
			if (error != kHaSuccess) continue;

			int tid = 0;
			#ifdef _OPENMP
			tid = omp_get_thread_num();
			#endif
			WorkspaceFP32 *ws = ws_arr[tid];

			int32_t off = sl_offsets[s];
			int32_t sz = sl_offsets[s + 1] - off;
			const int32_t *sl = &sl_indices[off];

			int rc;
			if (col_major) {
				// Col-major extraction + double→float conversion
				for (int32_t j = 0; j < sz; j++) {
					const double *src_x = &X_data[(size_t)sl[j] * nt];
					const double *src_y = &Y_data[(size_t)sl[j] * nt];
					float *dst_x = &ws->fX[j * nt];
					float *dst_y = &ws->fY[j * nt];
					for (int32_t i = 0; i < nt; i++) {
						dst_x[i] = (float)src_x[i];
						dst_y[i] = (float)src_y[i];
					}
				}
				rc = procrustes_ws_fp32_cm(ws, nt, sz, isReflection, isScaling);
			} else {
				extract_cols_d2f(X_data, nv, sl, sz, ws->fX, nt);
				extract_cols_d2f(Y_data, nv, sl, sz, ws->fY, nt);
				rc = procrustes_ws_fp32(ws, nt, sz, isReflection, isScaling);
			}
			if (rc != kHaSuccess) { error = rc; continue; }

			const double *w = &weights[off];
			if (col_major) {
				// fT is col-major: fT[i,j] = ws->fT[j * sz + i]
				for (int32_t i = 0; i < sz; i++) {
					int32_t row = sl[i];
					double wi = w[i];
					for (int32_t j = 0; j < sz; j++) {
						double val = (double)ws->fT[j * sz + i] * wi;
						#ifdef _OPENMP
						#pragma omp atomic
						#endif
						T_out[row * nv + sl[j]] += val;
					}
				}
			} else {
				for (int32_t i = 0; i < sz; i++) {
					int32_t row = sl[i];
					double wi = w[i];
					for (int32_t j = 0; j < sz; j++) {
						double val = (double)ws->fT[i * sz + j] * wi;
						#ifdef _OPENMP
						#pragma omp atomic
						#endif
						T_out[row * nv + sl[j]] += val;
					}
				}
			}
		}

		for (int t = 0; t < n_threads; t++) workspace_fp32_free(ws_arr[t]);
		free(ws_arr);
		free(weights);
		return error;
	}

	// ---- Metal batched + pipelined path ----
	#define METAL_BATCH_DENSE 256

	int32_t max_batch = METAL_BATCH_DENSE < count ? METAL_BATCH_DENSE : count;
	float **b_Xdata = (float **)malloc((size_t)max_batch * sizeof(float *));
	float **b_Ydata = (float **)malloc((size_t)max_batch * sizeof(float *));
	float **b_Adata = (float **)malloc((size_t)max_batch * sizeof(float *));
	int32_t *b_sizes = (int32_t *)malloc((size_t)max_batch * sizeof(int32_t));
	if (!b_Xdata || !b_Ydata || !b_Adata || !b_sizes) {
		free(b_Xdata); free(b_Ydata); free(b_Adata); free(b_sizes);
		free(weights);
		return kHaErrorAlloc;
	}

	int result = kHaSuccess;
	HaMetalBatch *pending = NULL;
	int32_t pending_start = 0, pending_count = 0;
	float **p_Xdata = NULL, **p_Ydata = NULL, **p_Adata = NULL;
	int32_t *p_sizes = NULL;

	for (int32_t batch_start = 0; batch_start < count; batch_start += METAL_BATCH_DENSE) {
		int32_t batch_end = batch_start + METAL_BATCH_DENSE;
		if (batch_end > count) batch_end = count;
		int32_t batch_count = batch_end - batch_start;

		for (int32_t i = 0; i < batch_count; i++) {
			int32_t s = batch_start + i;
			int32_t off = sl_offsets[s];
			int32_t sz = sl_offsets[s + 1] - off;
			const int32_t *sl = &sl_indices[off];
			b_sizes[i] = sz;

			// Metal pipeline expects row-major float data
			if (col_major) {
				// Extract from col-major double source into row-major float
				float *fxd = (float *)malloc((size_t)nt * sz * sizeof(float));
				float *fyd = (float *)malloc((size_t)nt * sz * sizeof(float));
				if (!fxd || !fyd) { free(fxd); free(fyd); result = kHaErrorAlloc; break; }
				for (int32_t j = 0; j < sz; j++) {
					const double *sx = &X_data[(size_t)sl[j] * nt];
					const double *sy = &Y_data[(size_t)sl[j] * nt];
					for (int32_t r = 0; r < nt; r++) {
						fxd[r * sz + j] = (float)sx[r];
						fyd[r * sz + j] = (float)sy[r];
					}
				}
				b_Xdata[i] = fxd;
				b_Ydata[i] = fyd;
			} else {
				TMat *lX = ha_mat_alloc(nt, sz);
				TMat *lY = ha_mat_alloc(nt, sz);
				if (!lX || !lY) { ha_mat_free(lX); ha_mat_free(lY); result = kHaErrorAlloc; break; }
				ha_mat_extract_cols(&X_mat, sl, sz, lX);
				ha_mat_extract_cols(&Y_mat, sl, sz, lY);

				TMatF *fX = ha_mat_to_float(lX);
				TMatF *fY = ha_mat_to_float(lY);
				ha_mat_free(lX);
				ha_mat_free(lY);
				if (!fX || !fY) { ha_matf_free(fX); ha_matf_free(fY); result = kHaErrorAlloc; break; }

				b_Xdata[i] = fX->data; free(fX);
				b_Ydata[i] = fY->data; free(fY);
			}
			b_Adata[i] = (float *)malloc((size_t)sz * sz * sizeof(float));
			if (!b_Adata[i]) { result = kHaErrorAlloc; break; }
		}
		if (result != kHaSuccess) break;

		HaMetalBatch *current = ha_metal_batch_submit(batch_count, nt, b_sizes,
		                                               b_Xdata, b_Ydata);
		if (!current) { result = kHaErrorInternal; break; }

		if (pending) {
			int rc = ha_metal_batch_wait(pending, p_Adata);
			ha_metal_batch_free(pending);
			pending = NULL;
			if (rc != kHaSuccess) { result = rc; break; }

			rc = metal_process_batch_dense(pending_start, pending_count, p_sizes,
			                               p_Xdata, p_Ydata, p_Adata,
			                               sl_indices, sl_offsets, weights,
			                               T_out, nv,
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
		if (rc != kHaSuccess) { result = rc; goto dense_cleanup; }

		rc = metal_process_batch_dense(pending_start, pending_count, p_sizes,
		                               p_Xdata, p_Ydata, p_Adata,
		                               sl_indices, sl_offsets, weights,
		                               T_out, nv,
		                               isReflection, isScaling, nt);
		if (rc != kHaSuccess) result = rc;
	}

dense_cleanup:
	if (pending) ha_metal_batch_free(pending);
	if (p_Xdata) {
		for (int32_t i = 0; i < pending_count; i++) {
			free(p_Xdata[i]); free(p_Ydata[i]); free(p_Adata[i]);
		}
		free(p_Xdata); free(p_Ydata); free(p_Adata); free(p_sizes);
	}
	free(b_Xdata); free(b_Ydata); free(b_Adata); free(b_sizes);
	free(weights);
	return result;
	#undef METAL_BATCH_DENSE
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
