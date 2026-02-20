/*
 * ha_common.h -- shared types, macros, and platform abstraction
 *
 * All hyperalignment C modules include this header. It provides:
 *   - Platform-conditional LAPACK/CBLAS includes
 *   - Dense matrix (TMat) and sparse matrix (TSparseCSC) types
 *   - Searchlight definition type (TSearchlights)
 *   - Allocation helpers and error codes
 */

#ifndef HA_COMMON_H
#define HA_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Platform-specific LAPACK/BLAS includes
#ifdef __APPLE__
	#define ACCELERATE_NEW_LAPACK
	#include <Accelerate/Accelerate.h>
	typedef __LAPACK_int ha_lapack_int;
#else
	#include <cblas.h>
	#include <lapack.h>
	typedef int ha_lapack_int;
#endif

// Error codes
enum {
	kHaSuccess       =  0,
	kHaErrorAlloc    = -1,
	kHaErrorSvd      = -2,
	kHaErrorArg      = -3,
	kHaErrorInternal = -4,
};

// Dense matrix: row-major, data[i * cols + j]
typedef struct {
	double *data;
	int32_t rows;
	int32_t cols;
} TMat;

// CSC sparse matrix (matches scipy.sparse.csc_matrix layout)
typedef struct {
	double *data;       // non-zero values, length nnz
	int32_t *indices;   // row indices, length nnz
	int32_t *indptr;    // column pointers, length cols + 1
	int32_t rows;
	int32_t cols;
	int64_t nnz;
} TSparseCSC;

// Searchlight definitions
typedef struct {
	int32_t **indices;  // vertex indices per searchlight
	double **dists;     // distances from center (NULL for uniform weighting)
	int32_t *sizes;     // number of vertices per searchlight
	int32_t count;      // number of searchlights
	double radius;      // searchlight radius (0 if uniform)
} TSearchlights;

// Backend selection for searchlight functions
typedef enum {
	kHaBackendCPU64 = 0,   // CPU double precision (default)
	kHaBackendCPU32 = 1,   // CPU single precision
	kHaBackendMetal = 2,   // Metal GPU (FP32)
} THaBackend;

// Element access macros
#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])
#define MATF_AT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

// Dense matrix: row-major, single precision
typedef struct {
	float *data;
	int32_t rows;
	int32_t cols;
} TMatF;

// ---- TMatF allocation helpers ----

static inline TMatF *ha_matf_alloc(int32_t rows, int32_t cols) {
	TMatF *m = (TMatF *)malloc(sizeof(TMatF));
	if (!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = (float *)malloc((size_t)rows * cols * sizeof(float));
	if (!m->data) { free(m); return NULL; }
	return m;
}

static inline TMatF *ha_matf_calloc(int32_t rows, int32_t cols) {
	TMatF *m = (TMatF *)malloc(sizeof(TMatF));
	if (!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = (float *)calloc((size_t)rows * cols, sizeof(float));
	if (!m->data) { free(m); return NULL; }
	return m;
}

static inline TMatF *ha_matf_copy(const TMatF *src) {
	TMatF *m = ha_matf_alloc(src->rows, src->cols);
	if (!m) return NULL;
	memcpy(m->data, src->data, (size_t)src->rows * src->cols * sizeof(float));
	return m;
}

static inline void ha_matf_free(TMatF *m) {
	if (m) {
		free(m->data);
		free(m);
	}
}

// ---- TMat allocation helpers ----

static inline TMat *ha_mat_alloc(int32_t rows, int32_t cols) {
	TMat *m = (TMat *)malloc(sizeof(TMat));
	if (!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = (double *)malloc((size_t)rows * cols * sizeof(double));
	if (!m->data) { free(m); return NULL; }
	return m;
}

static inline TMat *ha_mat_calloc(int32_t rows, int32_t cols) {
	TMat *m = (TMat *)malloc(sizeof(TMat));
	if (!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = (double *)calloc((size_t)rows * cols, sizeof(double));
	if (!m->data) { free(m); return NULL; }
	return m;
}

static inline TMat *ha_mat_copy(const TMat *src) {
	TMat *m = ha_mat_alloc(src->rows, src->cols);
	if (!m) return NULL;
	memcpy(m->data, src->data, (size_t)src->rows * src->cols * sizeof(double));
	return m;
}

static inline void ha_mat_zero(TMat *m) {
	memset(m->data, 0, (size_t)m->rows * m->cols * sizeof(double));
}

static inline void ha_mat_free(TMat *m) {
	if (m) {
		free(m->data);
		free(m);
	}
}

// ---- Precision conversion ----

static inline TMatF *ha_mat_to_float(const TMat *src) {
	TMatF *m = ha_matf_alloc(src->rows, src->cols);
	if (!m) return NULL;
	size_t n = (size_t)src->rows * src->cols;
	for (size_t i = 0; i < n; i++)
		m->data[i] = (float)src->data[i];
	return m;
}

static inline TMat *ha_matf_to_double(const TMatF *src) {
	TMat *m = ha_mat_alloc(src->rows, src->cols);
	if (!m) return NULL;
	size_t n = (size_t)src->rows * src->cols;
	for (size_t i = 0; i < n; i++)
		m->data[i] = (double)src->data[i];
	return m;
}

// ---- TSparseCSC helpers ----

static inline void ha_sparse_free(TSparseCSC *m) {
	if (m) {
		free(m->data);
		free(m->indices);
		free(m->indptr);
		free(m);
	}
}

static inline TSparseCSC *ha_sparse_copy(const TSparseCSC *src) {
	TSparseCSC *m = (TSparseCSC *)malloc(sizeof(TSparseCSC));
	if (!m) return NULL;
	m->rows = src->rows;
	m->cols = src->cols;
	m->nnz = src->nnz;
	m->data = (double *)malloc((size_t)src->nnz * sizeof(double));
	m->indices = (int32_t *)malloc((size_t)src->nnz * sizeof(int32_t));
	m->indptr = (int32_t *)malloc((size_t)(src->cols + 1) * sizeof(int32_t));
	if (!m->data || !m->indices || !m->indptr) {
		ha_sparse_free(m);
		return NULL;
	}
	memcpy(m->data, src->data, (size_t)src->nnz * sizeof(double));
	memcpy(m->indices, src->indices, (size_t)src->nnz * sizeof(int32_t));
	memcpy(m->indptr, src->indptr, (size_t)(src->cols + 1) * sizeof(int32_t));
	return m;
}

static inline void ha_sparse_zero_data(TSparseCSC *m) {
	memset(m->data, 0, (size_t)m->nnz * sizeof(double));
}

// ---- TSearchlights helpers ----

static inline void ha_searchlights_free(TSearchlights *sls) {
	if (sls) {
		for (int32_t i = 0; i < sls->count; i++) {
			free(sls->indices[i]);
			if (sls->dists) free(sls->dists[i]);
		}
		free(sls->indices);
		free(sls->dists);
		free(sls->sizes);
		free(sls);
	}
}

// ---- Utility: extract columns from a matrix ----

static inline void ha_mat_extract_cols(const TMat *src, const int32_t *col_idx,
                                       int32_t n_cols, TMat *dst) {
	for (int32_t i = 0; i < src->rows; i++) {
		for (int32_t j = 0; j < n_cols; j++) {
			dst->data[i * n_cols + j] = src->data[i * src->cols + col_idx[j]];
		}
	}
}

// ---- Utility: scatter columns back into a matrix (with additive weighting) ----

static inline void ha_mat_scatter_cols_add(TMat *dst, const int32_t *col_idx,
                                           int32_t n_cols, const TMat *src,
                                           const double *weights) {
	for (int32_t i = 0; i < src->rows; i++) {
		for (int32_t j = 0; j < n_cols; j++) {
			double w = weights ? weights[j] : 1.0;
			dst->data[i * dst->cols + col_idx[j]] += src->data[i * n_cols + j] * w;
		}
	}
}

#endif // HA_COMMON_H
