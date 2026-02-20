/*
 * ha_procrustes.h -- Orthogonal Procrustes algorithm
 */

#ifndef HA_PROCRUSTES_H
#define HA_PROCRUSTES_H

#include "ha_common.h"

/*
 * Orthogonal Procrustes: find T minimizing ||X @ T - Y||_F
 *
 * X: M x N, Y: M x N, T: N x N (pre-allocated).
 * Algorithm: A = X^T @ Y, SVD(A) = U s Vt, T = U @ Vt.
 * isReflection=true allows reflections; false forces det(T) > 0.
 * isScaling=true applies global scaling.
 *
 * X and Y are not modified.
 */
int ha_procrustes(const TMat *X, const TMat *Y, TMat *T,
                  bool isReflection, bool isScaling);

/*
 * Single-precision (FP32) Procrustes.
 * Same algorithm as ha_procrustes but uses float matrices and sgemm/sgesdd.
 */
int ha_procrustes_f32(const TMatF *X, const TMatF *Y, TMatF *T,
                      bool isReflection, bool isScaling);

/*
 * CPU polar decomposition via Newton iteration.
 * Given A = X^T @ Y (N x N), computes the orthogonal polar factor T.
 * Newton iteration: X_{k+1} = (X_k + X_k^{-T}) / 2, converges in ~6-10 iters.
 *
 * A: input N x N cross-correlation matrix (not modified)
 * T: output N x N orthogonal matrix (pre-allocated buffer, N*N floats)
 * isReflection: if false, forces det(T) > 0
 * isScaling: if true, scales T by trace(H) / (var(X)*M) where H = T^T @ A
 * X_data: original X matrix data (M x N, row-major), needed only if isScaling
 * M: number of rows in X, needed only if isScaling
 *
 * Returns kHaSuccess or kHaErrorInternal if A is singular.
 */
int ha_polar_newton(const float *A, float *T, int32_t N,
                    bool isReflection, bool isScaling,
                    const float *X_data, int32_t M);

#endif // HA_PROCRUSTES_H
