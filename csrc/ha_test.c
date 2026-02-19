/*
 * ha_test.c -- Test harness for hyperalignment C library
 */

#include "hyperalignment.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

// Test counters
static int n_tests = 0;
static int n_passed = 0;
static int n_failed = 0;

#define ASSERT_CLOSE(a, b, tol, msg) do { \
	n_tests++; \
	double _a = (a), _b = (b), _d = fabs(_a - _b); \
	if (_d > (tol)) { \
		fprintf(stderr, "  FAIL: %s: got %.12g, expected %.12g (diff %.2e)\n", \
		        (msg), _a, _b, _d); \
		n_failed++; \
	} else { n_passed++; } \
} while (0)

#define ASSERT_TRUE(cond, msg) do { \
	n_tests++; \
	if (!(cond)) { \
		fprintf(stderr, "  FAIL: %s\n", (msg)); \
		n_failed++; \
	} else { n_passed++; } \
} while (0)

// Deterministic PRNG for test data (splitmix64)
static uint64_t test_rng_state;

static void test_rng_seed(uint64_t seed) {
	test_rng_state = seed;
}

static uint64_t test_rng_next(void) {
	test_rng_state += 0x9e3779b97f4a7c15ULL;
	uint64_t z = test_rng_state;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

// Uniform [0, 1)
static double test_rand_uniform(void) {
	return (double)(test_rng_next() >> 11) / (double)(1ULL << 53);
}

// Standard normal via Box-Muller
static double test_randn(void) {
	double u1 = test_rand_uniform();
	double u2 = test_rand_uniform();
	if (u1 < 1e-15) u1 = 1e-15;
	return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Fill a matrix with random normal data
static void fill_randn(TMat *m) {
	for (int32_t i = 0; i < m->rows * m->cols; i++)
		m->data[i] = test_randn();
}

// ---- SVD Tests ----

static void test_svd_reconstruction(void) {
	printf("test_svd_reconstruction\n");
	test_rng_seed(42);

	int32_t M = 50, N = 30;
	int32_t K = (M < N) ? M : N;

	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	// Save a copy for verification
	TMat *X_orig = ha_mat_copy(X);

	TMat *U = ha_mat_alloc(M, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);

	int rc = ha_svd(X, U, s, Vt, false);
	ASSERT_TRUE(rc == kHaSuccess, "ha_svd returns success");

	// Verify singular values are non-negative and descending
	for (int32_t i = 0; i < K; i++)
		ASSERT_TRUE(s[i] >= 0, "singular value >= 0");
	for (int32_t i = 1; i < K; i++)
		ASSERT_TRUE(s[i] <= s[i - 1] + 1e-10, "singular values descending");

	// Reconstruct: X_rec = U * diag(s) * Vt
	// First: Us = U * diag(s)
	TMat *Us = ha_mat_alloc(M, K);
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < K; j++)
			Us->data[i * K + j] = U->data[i * K + j] * s[j];

	// X_rec = Us @ Vt
	TMat *X_rec = ha_mat_alloc(M, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, N, K, 1.0, Us->data, K, Vt->data, N, 0.0, X_rec->data, N);

	// Compare with original
	double max_err = 0.0;
	for (int32_t i = 0; i < M * N; i++) {
		double err = fabs(X_rec->data[i] - X_orig->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-10, "SVD reconstruction error < 1e-10");

	ha_mat_free(X);
	ha_mat_free(X_orig);
	ha_mat_free(U);
	free(s);
	ha_mat_free(Vt);
	ha_mat_free(Us);
	ha_mat_free(X_rec);
}

static void test_svd_wide_matrix(void) {
	printf("test_svd_wide_matrix\n");
	test_rng_seed(99);

	int32_t M = 30, N = 50;
	int32_t K = M;  // min(M, N)

	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);
	TMat *X_orig = ha_mat_copy(X);

	TMat *U = ha_mat_alloc(M, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);

	int rc = ha_svd(X, U, s, Vt, false);
	ASSERT_TRUE(rc == kHaSuccess, "ha_svd wide returns success");

	// Reconstruct
	TMat *Us = ha_mat_alloc(M, K);
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < K; j++)
			Us->data[i * K + j] = U->data[i * K + j] * s[j];

	TMat *X_rec = ha_mat_alloc(M, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, N, K, 1.0, Us->data, K, Vt->data, N, 0.0, X_rec->data, N);

	double max_err = 0.0;
	for (int32_t i = 0; i < M * N; i++) {
		double err = fabs(X_rec->data[i] - X_orig->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-10, "SVD wide reconstruction error < 1e-10");

	ha_mat_free(X); ha_mat_free(X_orig); ha_mat_free(U);
	free(s); ha_mat_free(Vt); ha_mat_free(Us); ha_mat_free(X_rec);
}

static void test_svd_mean_removal(void) {
	printf("test_svd_mean_removal\n");
	test_rng_seed(7);

	int32_t M = 40, N = 20;
	int32_t K = N;

	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	// Manual mean removal on a copy
	TMat *X_centered = ha_mat_copy(X);
	ha_remove_col_mean(X_centered);

	// SVD with mean removal
	TMat *X_copy = ha_mat_copy(X);
	TMat *U = ha_mat_alloc(M, K);
	double *s = (double *)malloc((size_t)K * sizeof(double));
	TMat *Vt = ha_mat_alloc(K, N);
	int rc = ha_svd(X_copy, U, s, Vt, true);
	ASSERT_TRUE(rc == kHaSuccess, "ha_svd with mean removal returns success");

	// Reconstruct and compare with centered X
	TMat *Us = ha_mat_alloc(M, K);
	for (int32_t i = 0; i < M; i++)
		for (int32_t j = 0; j < K; j++)
			Us->data[i * K + j] = U->data[i * K + j] * s[j];
	TMat *X_rec = ha_mat_alloc(M, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, N, K, 1.0, Us->data, K, Vt->data, N, 0.0, X_rec->data, N);

	double max_err = 0.0;
	for (int32_t i = 0; i < M * N; i++) {
		double err = fabs(X_rec->data[i] - X_centered->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-10, "SVD mean removal reconstruction error < 1e-10");

	ha_mat_free(X); ha_mat_free(X_centered); ha_mat_free(X_copy);
	ha_mat_free(U); free(s); ha_mat_free(Vt); ha_mat_free(Us); ha_mat_free(X_rec);
}

// ---- PCA Test ----

static void test_pca(void) {
	printf("test_pca\n");
	test_rng_seed(13);

	int32_t M = 40, N = 20;
	int32_t K = N;

	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	TMat *out = ha_mat_alloc(M, K);
	int rc = ha_pca(X, out, true);
	ASSERT_TRUE(rc == kHaSuccess, "ha_pca returns success");

	// Verify: out should be U * diag(s) from SVD of centered X
	// Just check that out has expected shape content (non-zero)
	double sum = 0.0;
	for (int32_t i = 0; i < M * K; i++)
		sum += out->data[i] * out->data[i];
	ASSERT_TRUE(sum > 0.0, "PCA output is non-zero");

	ha_mat_free(X);
	ha_mat_free(out);
}

// ---- Procrustes Tests ----

static void test_procrustes_identity(void) {
	printf("test_procrustes_identity\n");
	test_rng_seed(77);

	int32_t M = 50, N = 20;
	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	TMat *T = ha_mat_alloc(N, N);
	int rc = ha_procrustes(X, X, T, true, false);
	ASSERT_TRUE(rc == kHaSuccess, "procrustes identity returns success");

	// T should be close to identity
	for (int32_t i = 0; i < N; i++)
		for (int32_t j = 0; j < N; j++) {
			double expected = (i == j) ? 1.0 : 0.0;
			ASSERT_CLOSE(fabs(T->data[i * N + j]), fabs(expected), 1e-8,
			             "procrustes identity element");
		}

	ha_mat_free(X);
	ha_mat_free(T);
}

static void test_procrustes_known_rotation(void) {
	printf("test_procrustes_known_rotation\n");
	test_rng_seed(55);

	int32_t M = 50, N = 10;
	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	// Create a known rotation: Givens rotation in the (0,1) plane
	TMat *R = ha_mat_calloc(N, N);
	for (int32_t i = 0; i < N; i++)
		R->data[i * N + i] = 1.0;
	double angle = 0.7;
	R->data[0 * N + 0] = cos(angle);
	R->data[0 * N + 1] = -sin(angle);
	R->data[1 * N + 0] = sin(angle);
	R->data[1 * N + 1] = cos(angle);

	// Y = X @ R
	TMat *Y = ha_mat_alloc(M, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, N, N, 1.0, X->data, N, R->data, N, 0.0, Y->data, N);

	// Recover the rotation
	TMat *T = ha_mat_alloc(N, N);
	int rc = ha_procrustes(X, Y, T, true, false);
	ASSERT_TRUE(rc == kHaSuccess, "procrustes known rotation returns success");

	// X @ T should be close to Y
	TMat *XT = ha_mat_alloc(M, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, N, N, 1.0, X->data, N, T->data, N, 0.0, XT->data, N);

	double max_err = 0.0;
	for (int32_t i = 0; i < M * N; i++) {
		double err = fabs(XT->data[i] - Y->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-8, "procrustes known rotation alignment error < 1e-8");

	ha_mat_free(X); ha_mat_free(R); ha_mat_free(Y); ha_mat_free(T); ha_mat_free(XT);
}

static void test_procrustes_no_reflection(void) {
	printf("test_procrustes_no_reflection\n");
	test_rng_seed(33);

	int32_t M = 50, N = 10;
	TMat *X = ha_mat_alloc(M, N);
	TMat *Y = ha_mat_alloc(M, N);
	fill_randn(X);
	fill_randn(Y);

	TMat *T = ha_mat_alloc(N, N);
	int rc = ha_procrustes(X, Y, T, false, false);
	ASSERT_TRUE(rc == kHaSuccess, "procrustes no_reflection returns success");

	// Verify det(T) > 0 (proper rotation)
	// Use LU to compute determinant
	double *T_tmp = (double *)malloc((size_t)N * N * sizeof(double));
	ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
	memcpy(T_tmp, T->data, (size_t)N * N * sizeof(double));
	ha_lapack_int lN = N, info = 0;
	dgetrf_(&lN, &lN, T_tmp, &lN, ipiv, &info);

	double det_sign = 1.0;
	for (int32_t i = 0; i < N; i++) {
		if (T_tmp[i * N + i] < 0.0) det_sign = -det_sign;
		if (ipiv[i] != i + 1) det_sign = -det_sign;
	}
	ASSERT_TRUE(det_sign > 0.0, "det(T) > 0 when reflection disabled");

	free(T_tmp); free(ipiv);
	ha_mat_free(X); ha_mat_free(Y); ha_mat_free(T);
}

// ---- Ridge Test ----

static void test_ridge_small_alpha(void) {
	printf("test_ridge_small_alpha\n");
	test_rng_seed(101);

	// Generate Y = X @ true_betas so ridge with tiny alpha recovers them
	int32_t M = 50, N = 10, P = 5;
	TMat *X = ha_mat_alloc(M, N);
	TMat *true_betas = ha_mat_alloc(N, P);
	fill_randn(X);
	fill_randn(true_betas);

	TMat *Y = ha_mat_alloc(M, P);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            M, P, N, 1.0, X->data, N, true_betas->data, P, 0.0, Y->data, P);

	TMat *betas = ha_mat_alloc(N, P);
	int rc = ha_ridge(X, Y, 1e-6, betas);
	ASSERT_TRUE(rc == kHaSuccess, "ha_ridge returns success");

	// Recovered betas should be close to true_betas
	double max_err = 0.0;
	for (int32_t i = 0; i < N * P; i++) {
		double err = fabs(betas->data[i] - true_betas->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-6, "ridge recovers true betas");

	ha_mat_free(X); ha_mat_free(Y); ha_mat_free(true_betas);
	ha_mat_free(betas);
}

// ---- Sparse Tests ----

static void test_sparse_init(void) {
	printf("test_sparse_init\n");

	// Create simple searchlights: 2 overlapping searchlights on 5 vertices
	TSearchlights sls;
	sls.count = 2;
	sls.radius = 0;
	sls.dists = NULL;

	int32_t sl0[] = {0, 1, 2};
	int32_t sl1[] = {1, 2, 3, 4};
	int32_t *indices[] = {sl0, sl1};
	int32_t sizes[] = {3, 4};
	sls.indices = indices;
	sls.sizes = sizes;

	TSparseCSC *mat = ha_sparse_init(&sls, 5);
	ASSERT_TRUE(mat != NULL, "ha_sparse_init returns non-NULL");
	ASSERT_TRUE(mat->rows == 5, "sparse rows == 5");
	ASSERT_TRUE(mat->cols == 5, "sparse cols == 5");

	// Check nnz: sl0 contributes 3*3=9 pairs, sl1 contributes 4*4=16 pairs
	// Overlap pairs are (1,1),(1,2),(2,1),(2,2) = 4 shared pairs
	// Unique pairs: 9 + 16 - 4 = 21
	ASSERT_TRUE(mat->nnz == 21, "sparse nnz == 21");

	// All data should be zero
	double data_sum = 0.0;
	for (int64_t i = 0; i < mat->nnz; i++)
		data_sum += fabs(mat->data[i]);
	ASSERT_CLOSE(data_sum, 0.0, 1e-15, "sparse data all zeros");

	ha_sparse_free(mat);
}

static void test_sparse_scatter_add(void) {
	printf("test_sparse_scatter_add\n");

	TSearchlights sls;
	sls.count = 1;
	sls.radius = 0;
	sls.dists = NULL;

	int32_t sl0[] = {0, 1, 2};
	int32_t *indices[] = {sl0};
	int32_t sizes[] = {3};
	sls.indices = indices;
	sls.sizes = sizes;

	TSparseCSC *mat = ha_sparse_init(&sls, 3);
	ASSERT_TRUE(mat != NULL, "scatter_add sparse init");

	// Scatter-add a 3x3 identity matrix
	double local_T[] = {1, 0, 0,  0, 1, 0,  0, 0, 1};
	ha_sparse_scatter_add(mat, sl0, sl0, 3, 3, local_T, NULL);

	// Verify: diagonal entries should be 1, off-diagonal 0
	for (int32_t col = 0; col < 3; col++) {
		for (int64_t idx = mat->indptr[col]; idx < mat->indptr[col + 1]; idx++) {
			int32_t row = mat->indices[idx];
			double expected = (row == col) ? 1.0 : 0.0;
			ASSERT_CLOSE(mat->data[idx], expected, 1e-15, "scatter_add identity");
		}
	}

	ha_sparse_free(mat);
}

// ---- Searchlight Weights Test ----

static void test_searchlight_weights_uniform(void) {
	printf("test_searchlight_weights_uniform\n");

	// 3 overlapping searchlights on 6 vertices
	TSearchlights sls;
	sls.count = 3;
	sls.radius = 0;
	sls.dists = NULL;

	int32_t sl0[] = {0, 1, 2};
	int32_t sl1[] = {1, 2, 3, 4};
	int32_t sl2[] = {3, 4, 5};
	int32_t *sl_indices[] = {sl0, sl1, sl2};
	int32_t sl_sizes[] = {3, 4, 3};
	sls.indices = sl_indices;
	sls.sizes = sl_sizes;

	double *weights[3];
	int rc = ha_searchlight_weights(&sls, weights);
	ASSERT_TRUE(rc == kHaSuccess, "searchlight weights returns success");

	// Verify weights sum to 1 at each vertex
	double weight_sum[6] = {0};
	for (int32_t s = 0; s < 3; s++)
		for (int32_t i = 0; i < sl_sizes[s]; i++)
			weight_sum[sl_indices[s][i]] += weights[s][i];

	for (int32_t v = 0; v < 6; v++)
		ASSERT_CLOSE(weight_sum[v], 1.0, 1e-12, "weight sum == 1");

	for (int32_t s = 0; s < 3; s++) free(weights[s]);
}

// ---- Z-score Test ----

static void test_zscore(void) {
	printf("test_zscore\n");
	test_rng_seed(200);

	int32_t M = 100, N = 10;
	TMat *X = ha_mat_alloc(M, N);
	fill_randn(X);

	ha_zscore_columns(X);

	// Each column should have mean ~0 and std ~1
	for (int32_t j = 0; j < N; j++) {
		double sum = 0.0, sum2 = 0.0;
		for (int32_t i = 0; i < M; i++) {
			double v = X->data[i * N + j];
			sum += v;
			sum2 += v * v;
		}
		double mean = sum / M;
		double var = sum2 / M - mean * mean;
		ASSERT_CLOSE(mean, 0.0, 1e-12, "zscore mean == 0");
		ASSERT_CLOSE(var, 1.0, 1e-10, "zscore var == 1");
	}

	ha_mat_free(X);
}

// ---- Template Test ----

static void test_template_identical_subjects(void) {
	printf("test_template_identical_subjects\n");
	test_rng_seed(300);

	int32_t nt = 30, nv = 10, ns = 5;
	TMat *base = ha_mat_alloc(nt, nv);
	fill_randn(base);

	// All subjects identical
	const TMat *dms[5];
	TMat *copies[5];
	for (int32_t s = 0; s < ns; s++) {
		copies[s] = ha_mat_copy(base);
		dms[s] = copies[s];
	}

	TMat *tpl = ha_mat_alloc(nt, nv);
	int rc = ha_template_procrustes(dms, ns, true, false, false, 1, tpl);
	ASSERT_TRUE(rc == kHaSuccess, "template procrustes returns success");

	// Template should be close to the original (since all subjects are identical)
	// With isZscoreCommon=false, the template is the average = the input itself
	double max_err = 0.0;
	for (int32_t i = 0; i < nt * nv; i++) {
		double err = fabs(tpl->data[i] - base->data[i]);
		if (err > max_err) max_err = err;
	}
	ASSERT_TRUE(max_err < 1e-6, "template of identical subjects matches input");

	for (int32_t s = 0; s < ns; s++) ha_mat_free(copies[s]);
	ha_mat_free(base);
	ha_mat_free(tpl);
}

// ---- Ensemble Indices Test ----

static void test_ensemble_indices(void) {
	printf("test_ensemble_indices\n");

	int32_t **train_li, **test_li;
	int32_t *train_sz, *test_sz;
	int32_t n_splits;

	int rc = ha_ensemble_indices(100, 2, 5, 4, 4, 0, NULL,
	                             &train_li, &test_li, &train_sz, &test_sz, &n_splits);
	ASSERT_TRUE(rc == kHaSuccess, "ensemble_indices returns success");
	ASSERT_TRUE(n_splits == 10, "n_splits == 10 (2 perms * 5 folds)");

	// Check that train and test indices are within bounds
	for (int32_t i = 0; i < n_splits; i++) {
		ASSERT_TRUE(train_sz[i] > 0, "train_sz > 0");
		ASSERT_TRUE(test_sz[i] > 0, "test_sz > 0");
		for (int32_t j = 0; j < train_sz[i]; j++)
			ASSERT_TRUE(train_li[i][j] >= 0 && train_li[i][j] < 100, "train idx in bounds");
		for (int32_t j = 0; j < test_sz[i]; j++)
			ASSERT_TRUE(test_li[i][j] >= 0 && test_li[i][j] < 100, "test idx in bounds");
	}

	ha_ensemble_indices_free(train_li, test_li, train_sz, test_sz, n_splits);
}

// ---- Main ----

int main(void) {
	printf("Running hyperalignment C tests...\n\n");

	test_svd_reconstruction();
	test_svd_wide_matrix();
	test_svd_mean_removal();
	test_pca();
	test_procrustes_identity();
	test_procrustes_known_rotation();
	test_procrustes_no_reflection();
	test_ridge_small_alpha();
	test_sparse_init();
	test_sparse_scatter_add();
	test_searchlight_weights_uniform();
	test_zscore();
	test_template_identical_subjects();
	test_ensemble_indices();

	printf("\n%d tests: %d passed, %d failed\n", n_tests, n_passed, n_failed);
	return n_failed > 0 ? 1 : 0;
}
