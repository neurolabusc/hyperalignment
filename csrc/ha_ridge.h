/*
 * ha_ridge.h -- Ridge regression via SVD, grid search, ensemble
 */

#ifndef HA_RIDGE_H
#define HA_RIDGE_H

#include "ha_common.h"

/*
 * Ridge regression via SVD.
 * X: M x N, Y: M x P, alpha: regularization.
 * betas: N x P output (pre-allocated).
 * X and Y are not modified.
 */
int ha_ridge(const TMat *X, const TMat *Y, double alpha, TMat *betas);

/*
 * Ridge regression on a single target vector y (length M).
 * X: M x N, alpha: regularization.
 * betas_out: output vector of length N (pre-allocated).
 */
int ha_ridge_vec(const TMat *X, const double *y, double alpha, double *betas_out);

/*
 * Ridge grid search over alphas and PC counts.
 * X: M x N, y: vector of length M (single target).
 * alphas[n_alphas], npcs[n_npcs]: hyperparameter grids.
 * train_idx[train_n]: row indices for training (NULL = use all rows).
 * betas_out: flat array of size N * n_alphas * n_npcs (row-major 3D).
 *   betas_out[(v * n_alphas + a) * n_npcs + p] = beta for vertex v, alpha a, npc p.
 */
int ha_ridge_grid(const TMat *X, const double *y,
                  const double *alphas, int32_t n_alphas,
                  const int32_t *npcs, int32_t n_npcs,
                  const int32_t *train_idx, int32_t train_n,
                  double *betas_out);

/*
 * Ensemble ridge: cross-validated ridge with model averaging.
 * X: nt x nv, y: length nt.
 * Returns best weights (nv), predictions (nt), and best hyperparameters.
 */
int ha_ensemble_ridge(const TMat *X, const double *y,
                      const double *alphas, int32_t n_alphas,
                      const int32_t *npcs, int32_t n_npcs,
                      int32_t **train_idx_li, int32_t **test_idx_li,
                      const int32_t *train_sizes, const int32_t *test_sizes,
                      int32_t n_models,
                      double *weights_out, double *pred_out,
                      double *r2_out, double *alpha_out, int32_t *npc_out);

#endif // HA_RIDGE_H
