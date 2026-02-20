/*
 * ha_searchlight.h -- Searchlight weights, alignment loop
 */

#ifndef HA_SEARCHLIGHT_H
#define HA_SEARCHLIGHT_H

#include "ha_common.h"

/*
 * Set the number of OpenMP threads for searchlight loops.
 * Has no effect if the library was compiled without OpenMP.
 */
void ha_set_num_threads(int n);

/*
 * Get the current maximum number of OpenMP threads.
 * Returns 1 if compiled without OpenMP.
 */
int ha_get_num_threads(void);

/*
 * Compute searchlight combination weights (uniform or distance-based).
 * Weights are normalized to sum to 1.0 at each vertex.
 *
 * sls: searchlight definitions. If sls->dists is NULL, uniform weighting.
 * weights_out: pre-allocated array of sls->count pointers.
 *   Each weights_out[i] is allocated internally with sls->sizes[i] elements.
 *   Caller must free each weights_out[i].
 */
int ha_searchlight_weights(const TSearchlights *sls, double **weights_out);

/*
 * Searchlight Procrustes alignment.
 * Loops over all searchlights, computes local Procrustes, scatter-adds
 * into the global sparse matrix.
 *
 * X: nt x nv_X, Y: nt x nv_Y (not modified).
 * sls_X: searchlights for X.
 * sls_Y: searchlights for Y (NULL = same as sls_X).
 * mat: pre-initialized sparse matrix, accumulated in-place.
 * weights: per-searchlight vertex weights (NULL for unweighted).
 *   weights[i] has sls_X->sizes[i] elements.
 * isReflection, isScaling: Procrustes parameters.
 */
int ha_searchlight_procrustes(const TMat *X, const TMat *Y,
                              const TSearchlights *sls_X,
                              const TSearchlights *sls_Y,
                              TSparseCSC *mat,
                              double **weights,
                              bool isReflection, bool isScaling,
                              THaBackend backend);

/*
 * Searchlight Procrustes with flat-array input and dense output.
 * Weights are computed internally. No sparse structure is allocated.
 *
 * Input layout (flat concatenated arrays):
 *   sl_indices[sl_offsets[s] .. sl_offsets[s+1])  = vertex indices for searchlight s
 *   sl_dists[sl_offsets[s] .. sl_offsets[s+1])    = distances for searchlight s
 *   sl_offsets has (count + 1) elements; sl_offsets[0] = 0.
 *
 * X_data, Y_data: nt x nv matrices (not modified).
 *   col_major=false: row-major layout, X[i,j] = X_data[i*nv+j].
 *   col_major=true:  column-major (Fortran) layout, X[i,j] = X_data[j*nt+i].
 * T_out: pre-allocated nv x nv dense matrix (row-major), zero-initialized by caller.
 * sl_dists: NULL for uniform weighting.
 */
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
    bool col_major);

/*
 * Searchlight ridge alignment.
 * Same pattern as searchlight Procrustes but uses ridge regression.
 */
int ha_searchlight_ridge(const TMat *X, const TMat *Y,
                         const TSearchlights *sls_X,
                         const TSearchlights *sls_Y,
                         TSparseCSC *mat,
                         double **weights, double alpha);

#endif // HA_SEARCHLIGHT_H
