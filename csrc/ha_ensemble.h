/*
 * ha_ensemble.h -- Cross-validation index generation, ensemble searchlight
 */

#ifndef HA_ENSEMBLE_H
#define HA_ENSEMBLE_H

#include "ha_common.h"

/*
 * Generate cross-validation indices with temporal blocking.
 *
 * nt: number of timepoints
 * n_perms: number of permutations
 * n_folds: number of folds per permutation
 * blocksize: temporal block size
 * buffersize: exclusion buffer around test blocks
 * seed: random seed
 * mask: boolean mask of length nt (NULL = use all timepoints)
 *
 * Outputs (all allocated internally; caller must free):
 *   train_idx_li: array of n_splits pointers to train index arrays
 *   test_idx_li: array of n_splits pointers to test index arrays
 *   train_sizes: length of each train array
 *   test_sizes: length of each test array
 *   n_splits_out: n_perms * n_folds
 */
int ha_ensemble_indices(int32_t nt, int32_t n_perms, int32_t n_folds,
                        int32_t blocksize, int32_t buffersize, uint64_t seed,
                        const bool *mask,
                        int32_t ***train_idx_li, int32_t ***test_idx_li,
                        int32_t **train_sizes, int32_t **test_sizes,
                        int32_t *n_splits_out);

/*
 * Free the index arrays returned by ha_ensemble_indices.
 */
void ha_ensemble_indices_free(int32_t **train_idx_li, int32_t **test_idx_li,
                              int32_t *train_sizes, int32_t *test_sizes,
                              int32_t n_splits);

#endif // HA_ENSEMBLE_H
