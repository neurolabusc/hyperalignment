/*
 * ha_metal.h -- Metal GPU backend (Apple only)
 *
 * Provides GPU-accelerated Procrustes via polar decomposition using
 * Metal Performance Shaders (GEMM, LU, triangular solve). All GPU
 * operations use FP32.
 *
 * On non-Apple platforms, inline stubs return kHaErrorInternal and
 * ha_metal_available() returns false.
 */

#ifndef HA_METAL_H
#define HA_METAL_H

#include "ha_common.h"

#ifdef __APPLE__

/*
 * Initialize Metal device and command queue. Call once at startup.
 * Returns kHaSuccess or kHaErrorInternal if no Metal device found.
 */
int ha_metal_init(void);

/*
 * Release Metal resources. Safe to call even if ha_metal_init() was not called.
 */
void ha_metal_cleanup(void);

/*
 * Returns true if Metal GPU is initialized and available.
 */
bool ha_metal_available(void);

/*
 * GPU Procrustes via polar decomposition (Newton iteration).
 * Uses MPS GEMM for X^T @ Y, then CPU Newton iteration.
 *
 * X: M x N, Y: M x N, T: N x N (pre-allocated).
 */
int ha_procrustes_metal(const TMatF *X, const TMatF *Y, TMatF *T,
                        bool isReflection, bool isScaling);

/*
 * Batched GPU GEMM: A_i = X_i^T @ Y_i for i = 0..count-1.
 * All GEMMs are encoded into a single command buffer, submitted asynchronously.
 *
 * Opaque handle returned by ha_metal_batch_submit. Caller must:
 *   1. Call ha_metal_batch_wait() to block until GPU completes and read results
 *   2. Call ha_metal_batch_free() to release resources
 */
typedef struct HaMetalBatch HaMetalBatch;

HaMetalBatch *ha_metal_batch_submit(int32_t count, int32_t nt,
                                     const int32_t *sizes,
                                     float *const *X_data,
                                     float *const *Y_data);

int ha_metal_batch_wait(HaMetalBatch *batch, float **A_out);

void ha_metal_batch_free(HaMetalBatch *batch);

#else

// Stubs for non-Apple platforms
static inline int ha_metal_init(void) { return kHaErrorInternal; }
static inline void ha_metal_cleanup(void) {}
static inline bool ha_metal_available(void) { return false; }

static inline int ha_procrustes_metal(const TMatF *X, const TMatF *Y, TMatF *T,
                                      bool isReflection, bool isScaling) {
	(void)X; (void)Y; (void)T; (void)isReflection; (void)isScaling;
	return kHaErrorInternal;
}

typedef struct HaMetalBatch HaMetalBatch;
static inline HaMetalBatch *ha_metal_batch_submit(int32_t count, int32_t nt,
                                                   const int32_t *sizes,
                                                   float *const *X_data,
                                                   float *const *Y_data) {
	(void)count; (void)nt; (void)sizes; (void)X_data; (void)Y_data;
	return NULL;
}
static inline int ha_metal_batch_wait(HaMetalBatch *b, float **A) {
	(void)b; (void)A; return kHaErrorInternal;
}
static inline void ha_metal_batch_free(HaMetalBatch *b) { (void)b; }

#endif // __APPLE__

#endif // HA_METAL_H
