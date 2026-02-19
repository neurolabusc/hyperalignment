/*
 * ha_template.h -- Template construction (Procrustes, GPA, PCA)
 */

#ifndef HA_TEMPLATE_H
#define HA_TEMPLATE_H

#include "ha_common.h"

/*
 * Procrustes template: sequential alignment + iterative refinement.
 *
 * dms: array of n_subj pointers to TMat (each nt x nv).
 * isZscoreCommon: z-score the common space after each alignment.
 * level2_iter: number of leave-one-out refinement passes.
 * tpl_out: nt x nv output (pre-allocated).
 */
int ha_template_procrustes(const TMat **dms, int32_t n_subj,
                           bool isReflection, bool isScaling,
                           bool isZscoreCommon, int32_t level2_iter,
                           TMat *tpl_out);

/*
 * GPA template: generalized Procrustes analysis.
 * Starts from mean of all subjects rather than sequential.
 */
int ha_template_gpa(const TMat **dms, int32_t n_subj,
                    bool isReflection, bool isScaling,
                    bool isZscoreCommon, int32_t level2_iter,
                    TMat *tpl_out);

/*
 * PCA template: SVD of concatenated subject data.
 * tpl_out: nt x K output where K = min(nt, ns*nv).
 * npc_out: set to actual number of PCs returned.
 * isAdjustNs: divide by sqrt(n_subj) for variance normalization.
 */
int ha_template_pca(const TMat **dms, int32_t n_subj,
                    int32_t max_npc, bool isAdjustNs, bool isDemean,
                    TMat *tpl_out, int32_t *npc_out);

/*
 * Unified template interface.
 * kind: "procrustes", "pca", "gpa", "cls", "pcav1", "pcav2"
 * sl, sl_size: vertex subset (NULL, 0 = use all).
 * isCommonTopography: rotate template to match mean data topography.
 */
int ha_template(const TMat **dms, int32_t n_subj,
                const int32_t *sl, int32_t sl_size,
                const char *kind, int32_t max_npc,
                bool isCommonTopography, TMat *tpl_out);

#endif // HA_TEMPLATE_H
