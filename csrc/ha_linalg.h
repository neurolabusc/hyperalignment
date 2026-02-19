/*
 * ha_linalg.h -- SVD, PCA, column mean removal, column z-score
 */

#ifndef HA_LINALG_H
#define HA_LINALG_H

#include "ha_common.h"

/*
 * SVD with dgesdd fallback to dgesvd on failure.
 *
 * Decomposes X(M,N) = U(M,K) * diag(s)(K) * Vt(K,N) where K = min(M,N).
 * X is destroyed (LAPACK overwrites it). Caller must pre-allocate U, s, Vt.
 * If isRemoveMean is true, column means are subtracted before SVD.
 *
 * Returns kHaSuccess or kHaErrorSvd.
 */
int ha_svd(TMat *X, TMat *U, double *s, TMat *Vt, bool isRemoveMean);

/*
 * Convenience wrapper: copies X, allocates U/s/Vt, performs SVD.
 *
 * On success, caller owns *U_out (M x K), *s_out (K), *Vt_out (K x N)
 * where K = min(M, N). The copy of X is freed internally.
 * On failure, all intermediate allocations are cleaned up.
 */
int ha_svd_alloc(const TMat *X, TMat **U_out, double **s_out, TMat **Vt_out,
                 bool isRemoveMean);

/*
 * PCA via SVD: returns out = U * diag(s), shape M x K where K = min(M,N).
 * The input X is copied internally (not destroyed).
 * out must be pre-allocated as M x min(M,N).
 * If isRemoveMean is true, column means are subtracted.
 */
int ha_pca(const TMat *X, TMat *out, bool isRemoveMean);

/*
 * Subtract column means from X in-place.
 */
void ha_remove_col_mean(TMat *X);

/*
 * Column-wise z-score in-place (population std, ddof=0).
 * Zero-variance columns are set to 0.0.
 */
void ha_zscore_columns(TMat *X);

#endif // HA_LINALG_H
