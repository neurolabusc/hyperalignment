/*
 * ha_sparse.c -- CSC sparse matrix initialization and scatter-add
 *
 * Python reference (sparse.py):
 *   mat = lil_matrix((nv, nv))
 *   for sl in sls:
 *       mat[np.ix_(sl, sl)] = 1.
 *   mat = mat.tocsc()
 *   mat.data = np.zeros_like(mat.data)
 *
 * We enumerate all (row, col) pairs, sort by (col, row), deduplicate,
 * then build CSC arrays.
 */

#include "ha_sparse.h"

// Pair type for sorting
typedef struct {
	int32_t row;
	int32_t col;
} TPair;

static int pair_cmp(const void *a, const void *b) {
	const TPair *pa = (const TPair *)a;
	const TPair *pb = (const TPair *)b;
	if (pa->col != pb->col) return (pa->col < pb->col) ? -1 : 1;
	if (pa->row != pb->row) return (pa->row < pb->row) ? -1 : 1;
	return 0;
}

TSparseCSC *ha_sparse_init(const TSearchlights *sls, int32_t nv) {
	// Auto-detect nv if not provided
	if (nv <= 0) {
		int32_t max_idx = 0;
		for (int32_t s = 0; s < sls->count; s++)
			for (int32_t i = 0; i < sls->sizes[s]; i++)
				if (sls->indices[s][i] > max_idx)
					max_idx = sls->indices[s][i];
		nv = max_idx + 1;
	}

	// Count total pairs (with duplicates from overlapping searchlights)
	int64_t total_pairs = 0;
	for (int32_t s = 0; s < sls->count; s++)
		total_pairs += (int64_t)sls->sizes[s] * sls->sizes[s];

	TPair *pairs = (TPair *)malloc((size_t)total_pairs * sizeof(TPair));
	if (!pairs) return NULL;

	// Enumerate all (row, col) pairs
	int64_t idx = 0;
	for (int32_t s = 0; s < sls->count; s++) {
		int32_t *sl = sls->indices[s];
		int32_t sz = sls->sizes[s];
		for (int32_t i = 0; i < sz; i++)
			for (int32_t j = 0; j < sz; j++) {
				pairs[idx].row = sl[i];
				pairs[idx].col = sl[j];
				idx++;
			}
	}

	// Sort by (col, row)
	qsort(pairs, (size_t)total_pairs, sizeof(TPair), pair_cmp);

	// Count unique pairs
	int64_t nnz = 0;
	if (total_pairs > 0) {
		nnz = 1;
		for (int64_t i = 1; i < total_pairs; i++) {
			if (pairs[i].row != pairs[i - 1].row || pairs[i].col != pairs[i - 1].col)
				nnz++;
		}
	}

	// Allocate CSC structure
	TSparseCSC *mat = (TSparseCSC *)calloc(1, sizeof(TSparseCSC));
	if (!mat) { free(pairs); return NULL; }
	mat->rows = nv;
	mat->cols = nv;
	mat->nnz = nnz;
	mat->data = (double *)calloc((size_t)nnz, sizeof(double));
	mat->indices = (int32_t *)malloc((size_t)nnz * sizeof(int32_t));
	mat->indptr = (int32_t *)calloc((size_t)(nv + 1), sizeof(int32_t));
	if (!mat->data || !mat->indices || !mat->indptr) {
		ha_sparse_free(mat);
		free(pairs);
		return NULL;
	}

	// Build CSC arrays from sorted unique pairs
	int64_t write_idx = 0;
	if (total_pairs > 0) {
		mat->indices[0] = pairs[0].row;
		mat->indptr[pairs[0].col + 1]++;
		write_idx = 1;
		for (int64_t i = 1; i < total_pairs; i++) {
			if (pairs[i].row != pairs[i - 1].row || pairs[i].col != pairs[i - 1].col) {
				mat->indices[write_idx] = pairs[i].row;
				mat->indptr[pairs[i].col + 1]++;
				write_idx++;
			}
		}
	}
	free(pairs);

	// Convert column counts to cumulative pointers
	for (int32_t c = 1; c <= nv; c++)
		mat->indptr[c] += mat->indptr[c - 1];

	return mat;
}

// Binary search for row index within a CSC column
static inline int64_t sparse_find(const TSparseCSC *mat, int32_t row, int32_t col) {
	int64_t lo = mat->indptr[col];
	int64_t hi = mat->indptr[col + 1];
	while (lo < hi) {
		int64_t mid = (lo + hi) / 2;
		if (mat->indices[mid] < row) lo = mid + 1;
		else hi = mid;
	}
	if (lo < mat->indptr[col + 1] && mat->indices[lo] == row)
		return lo;
	return -1;  // not found (should not happen if sparsity pattern is correct)
}

void ha_sparse_scatter_add(TSparseCSC *mat,
                           const int32_t *sl_rows, const int32_t *sl_cols,
                           int32_t sl_n_rows, int32_t sl_n_cols,
                           const double *local_T, const double *weights) {
	for (int32_t j = 0; j < sl_n_cols; j++) {
		int32_t col = sl_cols[j];
		for (int32_t i = 0; i < sl_n_rows; i++) {
			double w = weights ? weights[i] : 1.0;
			double val = local_T[i * sl_n_cols + j] * w;
			int64_t pos = sparse_find(mat, sl_rows[i], col);
			if (pos >= 0)
				mat->data[pos] += val;
		}
	}
}
