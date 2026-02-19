/*
 * ha_ensemble.c -- Cross-validation index generation
 *
 * Python reference (ensemble.py):
 *   Block-permuted CV splits with temporal buffer zones.
 *   Uses xoshiro256** PRNG (not matching Python's PCG64, but correct structure).
 */

#include "ha_ensemble.h"

// xoshiro256** PRNG
typedef struct {
	uint64_t s[4];
} TRng;

static inline uint64_t rotl(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(TRng *rng) {
	uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
	uint64_t t = rng->s[1] << 17;
	rng->s[2] ^= rng->s[0];
	rng->s[3] ^= rng->s[1];
	rng->s[1] ^= rng->s[2];
	rng->s[0] ^= rng->s[3];
	rng->s[2] ^= t;
	rng->s[3] = rotl(rng->s[3], 45);
	return result;
}

// Initialize RNG from a single seed using splitmix64
static void rng_seed(TRng *rng, uint64_t seed) {
	for (int i = 0; i < 4; i++) {
		seed += 0x9e3779b97f4a7c15ULL;
		uint64_t z = seed;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		rng->s[i] = z ^ (z >> 31);
	}
}

// Random integer in [0, n)
static int32_t rng_int(TRng *rng, int32_t n) {
	return (int32_t)(rng_next(rng) % (uint64_t)n);
}

// Fisher-Yates shuffle
static void shuffle(TRng *rng, int32_t *arr, int32_t n) {
	for (int32_t i = n - 1; i > 0; i--) {
		int32_t j = rng_int(rng, i + 1);
		int32_t tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

int ha_ensemble_indices(int32_t nt, int32_t n_perms, int32_t n_folds,
                        int32_t blocksize, int32_t buffersize, uint64_t seed,
                        const bool *mask,
                        int32_t ***train_idx_li_out, int32_t ***test_idx_li_out,
                        int32_t **train_sizes_out, int32_t **test_sizes_out,
                        int32_t *n_splits_out) {
	TRng rng;
	rng_seed(&rng, seed);

	int32_t n_blocks = nt / blocksize;
	int32_t remainder = nt % blocksize;
	int32_t n_splits = n_perms * n_folds;

	*n_splits_out = n_splits;

	int32_t **train_li = (int32_t **)calloc((size_t)n_splits, sizeof(int32_t *));
	int32_t **test_li = (int32_t **)calloc((size_t)n_splits, sizeof(int32_t *));
	int32_t *train_sz = (int32_t *)calloc((size_t)n_splits, sizeof(int32_t));
	int32_t *test_sz = (int32_t *)calloc((size_t)n_splits, sizeof(int32_t));
	if (!train_li || !test_li || !train_sz || !test_sz) {
		free(train_li); free(test_li); free(train_sz); free(test_sz);
		return kHaErrorAlloc;
	}

	// Cumulative mask mapping (for masked timepoints)
	int32_t *mapping = NULL;
	if (mask) {
		mapping = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
		if (!mapping) {
			free(train_li); free(test_li); free(train_sz); free(test_sz);
			return kHaErrorAlloc;
		}
		int32_t cum = 0;
		for (int32_t i = 0; i < nt; i++) {
			if (mask[i]) mapping[i] = cum++;
			else mapping[i] = -1;
		}
	}

	int32_t *block_order = (int32_t *)malloc((size_t)n_blocks * sizeof(int32_t));
	int32_t padded = nt + buffersize * 2;
	bool *train_mask = (bool *)malloc((size_t)padded * sizeof(bool));
	bool *test_mask = (bool *)malloc((size_t)padded * sizeof(bool));
	if (!block_order || !train_mask || !test_mask) {
		free(block_order); free(train_mask); free(test_mask); free(mapping);
		free(train_li); free(test_li); free(train_sz); free(test_sz);
		return kHaErrorAlloc;
	}

	int32_t split_idx = 0;
	for (int32_t perm = 0; perm < n_perms; perm++) {
		int32_t shift = rng_int(&rng, remainder + 1);

		// Permute block order
		for (int32_t i = 0; i < n_blocks; i++) block_order[i] = i;
		shuffle(&rng, block_order, n_blocks);

		// Split blocks into folds
		int32_t fold_start = 0;
		for (int32_t fold = 0; fold < n_folds; fold++) {
			int32_t fold_end = (int32_t)((int64_t)(fold + 1) * n_blocks / n_folds);
			int32_t n_test_blocks = fold_end - fold_start;

			// Build test indices from test blocks
			int32_t n_test_raw = n_test_blocks * blocksize;
			int32_t *test_raw = (int32_t *)malloc((size_t)n_test_raw * sizeof(int32_t));
			if (!test_raw) goto alloc_fail;
			int32_t ti = 0;
			for (int32_t b = fold_start; b < fold_end; b++) {
				for (int32_t k = 0; k < blocksize; k++) {
					int32_t idx = block_order[b] * blocksize + k + shift;
					if (idx < nt)
						test_raw[ti++] = idx;
				}
			}
			n_test_raw = ti;

			// Build train mask: exclude test indices and buffer
			for (int32_t i = 0; i < padded; i++) train_mask[i] = true;
			for (int32_t i = 0; i < n_test_raw; i++) {
				int32_t center = test_raw[i] + buffersize;
				for (int32_t b = -buffersize; b <= buffersize; b++) {
					int32_t pos = center + b;
					if (pos >= 0 && pos < padded)
						train_mask[pos] = false;
				}
			}
			// Apply external mask
			if (mask) {
				for (int32_t i = 0; i < nt; i++) {
					if (!mask[i])
						train_mask[i + buffersize] = false;
				}
			}

			// Collect valid training indices
			int32_t n_valid_train = 0;
			for (int32_t i = 0; i < nt; i++)
				if (train_mask[i + buffersize])
					n_valid_train++;

			int32_t *valid_train = (int32_t *)malloc((size_t)n_valid_train * sizeof(int32_t));
			if (!valid_train) { free(test_raw); goto alloc_fail; }
			ti = 0;
			for (int32_t i = 0; i < nt; i++)
				if (train_mask[i + buffersize])
					valid_train[ti++] = i;

			// Bootstrap sample of size nt from valid training indices
			int32_t *train_idx = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
			if (!train_idx) { free(test_raw); free(valid_train); goto alloc_fail; }
			if (n_valid_train > 0) {
				for (int32_t i = 0; i < nt; i++)
					train_idx[i] = valid_train[rng_int(&rng, n_valid_train)];
			}

			// Build test mask: exclude train indices and buffer
			for (int32_t i = 0; i < padded; i++) test_mask[i] = true;
			// Get unique train indices
			bool *used = (bool *)calloc((size_t)nt, sizeof(bool));
			if (!used) { free(test_raw); free(valid_train); free(train_idx); goto alloc_fail; }
			for (int32_t i = 0; i < nt; i++)
				used[train_idx[i]] = true;
			for (int32_t i = 0; i < nt; i++) {
				if (used[i]) {
					int32_t center = i + buffersize;
					for (int32_t b = -buffersize; b <= buffersize; b++) {
						int32_t pos = center + b;
						if (pos >= 0 && pos < padded)
							test_mask[pos] = false;
					}
				}
			}
			free(used);
			if (mask) {
				for (int32_t i = 0; i < nt; i++)
					if (!mask[i])
						test_mask[i + buffersize] = false;
			}

			// Collect test indices
			int32_t n_test = 0;
			for (int32_t i = 0; i < nt; i++)
				if (test_mask[i + buffersize])
					n_test++;
			int32_t *test_idx = (int32_t *)malloc((size_t)n_test * sizeof(int32_t));
			if (!test_idx) { free(test_raw); free(valid_train); free(train_idx); goto alloc_fail; }
			ti = 0;
			for (int32_t i = 0; i < nt; i++)
				if (test_mask[i + buffersize])
					test_idx[ti++] = i;

			// Apply mask mapping
			if (mask) {
				for (int32_t i = 0; i < nt; i++)
					train_idx[i] = mapping[train_idx[i]];
				for (int32_t i = 0; i < n_test; i++)
					test_idx[i] = mapping[test_idx[i]];
			}

			train_li[split_idx] = train_idx;
			test_li[split_idx] = test_idx;
			train_sz[split_idx] = nt;
			test_sz[split_idx] = n_test;
			split_idx++;

			free(test_raw);
			free(valid_train);
			fold_start = fold_end;
		}
	}

	free(block_order);
	free(train_mask);
	free(test_mask);
	free(mapping);

	*train_idx_li_out = train_li;
	*test_idx_li_out = test_li;
	*train_sizes_out = train_sz;
	*test_sizes_out = test_sz;
	return kHaSuccess;

alloc_fail:
	free(block_order);
	free(train_mask);
	free(test_mask);
	free(mapping);
	ha_ensemble_indices_free(train_li, test_li, train_sz, test_sz, n_splits);
	return kHaErrorAlloc;
}

void ha_ensemble_indices_free(int32_t **train_idx_li, int32_t **test_idx_li,
                              int32_t *train_sizes, int32_t *test_sizes,
                              int32_t n_splits) {
	if (train_idx_li) {
		for (int32_t i = 0; i < n_splits; i++)
			free(train_idx_li[i]);
		free(train_idx_li);
	}
	if (test_idx_li) {
		for (int32_t i = 0; i < n_splits; i++)
			free(test_idx_li[i]);
		free(test_idx_li);
	}
	free(train_sizes);
	free(test_sizes);
}
