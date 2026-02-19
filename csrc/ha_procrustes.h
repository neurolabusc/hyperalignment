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

#endif // HA_PROCRUSTES_H
