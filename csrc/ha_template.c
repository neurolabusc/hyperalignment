/*
 * ha_template.c -- Template construction (Procrustes, GPA, PCA)
 *
 * Python reference (local_template.py):
 *   Procrustes: sequential alignment to running average + level2 refinement
 *   GPA: start from mean, iterative alignment
 *   PCA: SVD of concatenated data
 */

#include "ha_template.h"
#include "ha_linalg.h"
#include "ha_procrustes.h"

int ha_template_procrustes(const TMat **dms, int32_t n_subj,
                           bool isReflection, bool isScaling,
                           bool isZscoreCommon, int32_t level2_iter,
                           TMat *tpl_out) {
	int32_t nt = dms[0]->rows;
	int32_t nv = dms[0]->cols;
	int rc;

	// common_space = copy of dms[0]
	TMat *common = ha_mat_copy(dms[0]);
	if (!common) return kHaErrorAlloc;

	// aligned_dss[i] = aligned version of dms[i]
	TMat **aligned = (TMat **)malloc((size_t)n_subj * sizeof(TMat *));
	if (!aligned) { ha_mat_free(common); return kHaErrorAlloc; }
	aligned[0] = ha_mat_copy(dms[0]);
	if (!aligned[0]) { ha_mat_free(common); free(aligned); return kHaErrorAlloc; }

	TMat *T = ha_mat_alloc(nv, nv);
	if (!T) {
		ha_mat_free(common); ha_mat_free(aligned[0]); free(aligned);
		return kHaErrorAlloc;
	}

	// Sequential alignment: each subject aligns to running average
	for (int32_t s = 1; s < n_subj; s++) {
		rc = ha_procrustes(dms[s], common, T, isReflection, isScaling);
		if (rc != kHaSuccess) goto cleanup;

		// aligned[s] = dms[s] @ T
		aligned[s] = ha_mat_alloc(nt, nv);
		if (!aligned[s]) { rc = kHaErrorAlloc; goto cleanup; }
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            nt, nv, nv, 1.0, dms[s]->data, nv, T->data, nv,
		            0.0, aligned[s]->data, nv);

		if (isZscoreCommon)
			ha_zscore_columns(aligned[s]);

		// common = (common + aligned[s]) * 0.5
		for (int32_t i = 0; i < nt * nv; i++)
			common->data[i] = (common->data[i] + aligned[s]->data[i]) * 0.5;

		if (isZscoreCommon)
			ha_zscore_columns(common);
	}

	// Level-2 iterative refinement: leave-one-out re-alignment
	for (int32_t iter = 0; iter < level2_iter; iter++) {
		// Compute sum of all aligned subjects
		ha_mat_zero(common);
		for (int32_t s = 0; s < n_subj; s++)
			for (int32_t i = 0; i < nt * nv; i++)
				common->data[i] += aligned[s]->data[i];

		for (int32_t s = 0; s < n_subj; s++) {
			// reference = (sum - aligned[s]) / (n_subj - 1)
			TMat *ref = ha_mat_alloc(nt, nv);
			if (!ref) { rc = kHaErrorAlloc; goto cleanup; }
			double inv_ns = 1.0 / (n_subj - 1);
			for (int32_t i = 0; i < nt * nv; i++)
				ref->data[i] = (common->data[i] - aligned[s]->data[i]) * inv_ns;

			if (isZscoreCommon)
				ha_zscore_columns(ref);

			rc = ha_procrustes(dms[s], ref, T, isReflection, isScaling);
			ha_mat_free(ref);
			if (rc != kHaSuccess) goto cleanup;

			// aligned[s] = dms[s] @ T
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			            nt, nv, nv, 1.0, dms[s]->data, nv, T->data, nv,
			            0.0, aligned[s]->data, nv);
		}
	}

	// Final template = sum of aligned, then z-score or average
	ha_mat_zero(tpl_out);
	for (int32_t s = 0; s < n_subj; s++)
		for (int32_t i = 0; i < nt * nv; i++)
			tpl_out->data[i] += aligned[s]->data[i];

	if (isZscoreCommon) {
		ha_zscore_columns(tpl_out);
	} else {
		double inv_ns = 1.0 / n_subj;
		for (int32_t i = 0; i < nt * nv; i++)
			tpl_out->data[i] *= inv_ns;
	}

	rc = kHaSuccess;

cleanup:
	ha_mat_free(T);
	ha_mat_free(common);
	for (int32_t s = 0; s < n_subj; s++)
		if (aligned[s]) ha_mat_free(aligned[s]);
	free(aligned);
	return rc;
}

int ha_template_gpa(const TMat **dms, int32_t n_subj,
                    bool isReflection, bool isScaling,
                    bool isZscoreCommon, int32_t level2_iter,
                    TMat *tpl_out) {
	int32_t nt = dms[0]->rows;
	int32_t nv = dms[0]->cols;
	int rc;

	// common_space = mean of all subjects
	TMat *common = ha_mat_calloc(nt, nv);
	if (!common) return kHaErrorAlloc;
	for (int32_t s = 0; s < n_subj; s++)
		for (int32_t i = 0; i < nt * nv; i++)
			common->data[i] += dms[s]->data[i];
	double inv_ns = 1.0 / n_subj;
	for (int32_t i = 0; i < nt * nv; i++)
		common->data[i] *= inv_ns;

	if (isZscoreCommon)
		ha_zscore_columns(common);

	// Initial alignment: each subject to common
	TMat **aligned = (TMat **)calloc((size_t)n_subj, sizeof(TMat *));
	TMat *T = ha_mat_alloc(nv, nv);
	if (!aligned || !T) {
		ha_mat_free(common); free(aligned); ha_mat_free(T);
		return kHaErrorAlloc;
	}

	for (int32_t s = 0; s < n_subj; s++) {
		rc = ha_procrustes(dms[s], common, T, isReflection, isScaling);
		if (rc != kHaSuccess) goto cleanup;
		aligned[s] = ha_mat_alloc(nt, nv);
		if (!aligned[s]) { rc = kHaErrorAlloc; goto cleanup; }
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		            nt, nv, nv, 1.0, dms[s]->data, nv, T->data, nv,
		            0.0, aligned[s]->data, nv);
		if (isZscoreCommon)
			ha_zscore_columns(aligned[s]);
	}

	// Level-2 iterative refinement (same as Procrustes template)
	for (int32_t iter = 0; iter < level2_iter; iter++) {
		ha_mat_zero(common);
		for (int32_t s = 0; s < n_subj; s++)
			for (int32_t i = 0; i < nt * nv; i++)
				common->data[i] += aligned[s]->data[i];

		for (int32_t s = 0; s < n_subj; s++) {
			TMat *ref = ha_mat_alloc(nt, nv);
			if (!ref) { rc = kHaErrorAlloc; goto cleanup; }
			double inv = 1.0 / (n_subj - 1);
			for (int32_t i = 0; i < nt * nv; i++)
				ref->data[i] = (common->data[i] - aligned[s]->data[i]) * inv;
			if (isZscoreCommon)
				ha_zscore_columns(ref);

			rc = ha_procrustes(dms[s], ref, T, isReflection, isScaling);
			ha_mat_free(ref);
			if (rc != kHaSuccess) goto cleanup;

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			            nt, nv, nv, 1.0, dms[s]->data, nv, T->data, nv,
			            0.0, aligned[s]->data, nv);
		}
	}

	// Final template
	ha_mat_zero(tpl_out);
	for (int32_t s = 0; s < n_subj; s++)
		for (int32_t i = 0; i < nt * nv; i++)
			tpl_out->data[i] += aligned[s]->data[i];

	if (isZscoreCommon) {
		ha_zscore_columns(tpl_out);
	} else {
		for (int32_t i = 0; i < nt * nv; i++)
			tpl_out->data[i] /= n_subj;
	}

	rc = kHaSuccess;

cleanup:
	ha_mat_free(T);
	ha_mat_free(common);
	if (aligned) {
		for (int32_t s = 0; s < n_subj; s++)
			ha_mat_free(aligned[s]);
		free(aligned);
	}
	return rc;
}

int ha_template_pca(const TMat **dms, int32_t n_subj,
                    int32_t max_npc, bool isAdjustNs, bool isDemean,
                    TMat *tpl_out, int32_t *npc_out) {
	int32_t nt = dms[0]->rows;
	int32_t nv = dms[0]->cols;
	int32_t nv_concat = n_subj * nv;

	// Concatenate: X_concat(nt, ns*nv) where subject s fills columns [s*nv, (s+1)*nv)
	TMat *X = ha_mat_alloc(nt, nv_concat);
	if (!X) return kHaErrorAlloc;
	for (int32_t s = 0; s < n_subj; s++) {
		for (int32_t i = 0; i < nt; i++) {
			memcpy(&X->data[i * nv_concat + s * nv],
			       &dms[s]->data[i * nv],
			       (size_t)nv * sizeof(double));
		}
	}

	int32_t K = (nt < nv_concat) ? nt : nv_concat;
	if (max_npc > 0 && max_npc < K)
		K = max_npc;

	// SVD of concatenated data
	TMat *Xc = ha_mat_copy(X);
	ha_mat_free(X);
	if (!Xc) return kHaErrorAlloc;

	int32_t K_full = (nt < nv_concat) ? nt : nv_concat;
	TMat *U = ha_mat_alloc(nt, K_full);
	double *s_vals = (double *)malloc((size_t)K_full * sizeof(double));
	TMat *Vt = ha_mat_alloc(K_full, nv_concat);
	if (!U || !s_vals || !Vt) {
		ha_mat_free(Xc); ha_mat_free(U); free(s_vals); ha_mat_free(Vt);
		return kHaErrorAlloc;
	}

	int rc = ha_svd(Xc, U, s_vals, Vt, isDemean);
	ha_mat_free(Xc);
	if (rc != kHaSuccess) {
		ha_mat_free(U); free(s_vals); ha_mat_free(Vt);
		return rc;
	}

	// XX = U[:, :K] * s[:K]
	double scale = isAdjustNs ? (1.0 / sqrt((double)n_subj)) : 1.0;
	for (int32_t i = 0; i < nt; i++)
		for (int32_t j = 0; j < K; j++)
			tpl_out->data[i * K + j] = U->data[i * K_full + j] * s_vals[j] * scale;

	*npc_out = K;

	ha_mat_free(U);
	free(s_vals);
	ha_mat_free(Vt);
	return kHaSuccess;
}

int ha_template(const TMat **dms, int32_t n_subj,
                const int32_t *sl, int32_t sl_size,
                const char *kind, int32_t max_npc,
                bool isCommonTopography, TMat *tpl_out) {
	int32_t nt = dms[0]->rows;
	int32_t nv = sl ? sl_size : dms[0]->cols;
	int rc;

	// Extract submatrices if sl is specified
	const TMat **dms_local = dms;
	TMat **dms_sub = NULL;
	if (sl) {
		dms_sub = (TMat **)malloc((size_t)n_subj * sizeof(TMat *));
		if (!dms_sub) return kHaErrorAlloc;
		for (int32_t s = 0; s < n_subj; s++) {
			dms_sub[s] = ha_mat_alloc(nt, sl_size);
			if (!dms_sub[s]) {
				for (int32_t j = 0; j < s; j++) ha_mat_free(dms_sub[j]);
				free(dms_sub);
				return kHaErrorAlloc;
			}
			ha_mat_extract_cols(dms[s], sl, sl_size, dms_sub[s]);
		}
		dms_local = (const TMat **)dms_sub;
	}

	if (strcmp(kind, "procrustes") == 0 || strcmp(kind, "cls") == 0) {
		rc = ha_template_procrustes(dms_local, n_subj, true, false, true, 1, tpl_out);
	} else if (strcmp(kind, "gpa") == 0) {
		rc = ha_template_gpa(dms_local, n_subj, true, false, true, 1, tpl_out);
	} else if (strcmp(kind, "pca") == 0) {
		int32_t npc_actual;
		rc = ha_template_pca(dms_local, n_subj, max_npc, true, true, tpl_out, &npc_actual);
	} else {
		rc = kHaErrorArg;
	}

	// Common topography rotation
	if (rc == kHaSuccess && isCommonTopography) {
		// Tile template ns times and align to concatenated dms
		int32_t ns = n_subj;
		TMat *tiled = ha_mat_alloc(ns * nt, nv);
		TMat *concat = ha_mat_alloc(ns * nt, nv);
		TMat *T = ha_mat_alloc(nv, nv);
		if (!tiled || !concat || !T) {
			ha_mat_free(tiled); ha_mat_free(concat); ha_mat_free(T);
			rc = kHaErrorAlloc;
		} else {
			for (int32_t s = 0; s < ns; s++) {
				memcpy(&tiled->data[(size_t)s * nt * nv], tpl_out->data,
				       (size_t)nt * nv * sizeof(double));
				memcpy(&concat->data[(size_t)s * nt * nv], dms_local[s]->data,
				       (size_t)nt * nv * sizeof(double));
			}
			rc = ha_procrustes(tiled, concat, T, true, false);
			if (rc == kHaSuccess) {
				// tpl_out = tpl_out @ T
				TMat *tmp = ha_mat_alloc(nt, nv);
				if (!tmp) {
					rc = kHaErrorAlloc;
				} else {
					cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					            nt, nv, nv, 1.0, tpl_out->data, nv, T->data, nv,
					            0.0, tmp->data, nv);
					memcpy(tpl_out->data, tmp->data, (size_t)nt * nv * sizeof(double));
					ha_mat_free(tmp);
				}
			}
			ha_mat_free(tiled);
			ha_mat_free(concat);
			ha_mat_free(T);
		}
	}

	if (dms_sub) {
		for (int32_t s = 0; s < n_subj; s++)
			ha_mat_free(dms_sub[s]);
		free(dms_sub);
	}

	return rc;
}
