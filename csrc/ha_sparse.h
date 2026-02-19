/*
 * ha_sparse.h -- CSC sparse matrix initialization and scatter-add
 */

#ifndef HA_SPARSE_H
#define HA_SPARSE_H

#include "ha_common.h"

/*
 * Initialize a CSC sparse matrix from searchlight sparsity pattern.
 * For each searchlight sl, all (sl[i], sl[j]) pairs are non-zero positions.
 * The data array is initialized to zeros (sparsity pattern only).
 *
 * sls: searchlight definitions
 * nv: number of vertices (0 = auto-detect from max index in sls)
 *
 * Returns allocated TSparseCSC or NULL on failure.
 */
TSparseCSC *ha_sparse_init(const TSearchlights *sls, int32_t nv);

/*
 * Scatter-add a dense local transformation into the sparse matrix.
 *
 * mat: global sparse matrix (modified in-place)
 * sl_rows: vertex indices for rows (length sl_n_rows)
 * sl_cols: vertex indices for columns (length sl_n_cols)
 * sl_n_rows, sl_n_cols: dimensions of the local transformation
 * local_T: dense (sl_n_rows x sl_n_cols) transformation to scatter
 * weights: per-row weights (length sl_n_rows), or NULL for unweighted
 *
 * For each (i, j), adds local_T[i,j] * weights[i] into mat[sl_rows[i], sl_cols[j]].
 * Uses binary search to find positions in CSC data[].
 */
void ha_sparse_scatter_add(TSparseCSC *mat,
                           const int32_t *sl_rows, const int32_t *sl_cols,
                           int32_t sl_n_rows, int32_t sl_n_cols,
                           const double *local_T, const double *weights);

#endif // HA_SPARSE_H
