/*
 * ha_searchlight.h -- Searchlight weights, alignment loop
 */

#ifndef HA_SEARCHLIGHT_H
#define HA_SEARCHLIGHT_H

#include "ha_common.h"

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
                              bool isReflection, bool isScaling);

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
