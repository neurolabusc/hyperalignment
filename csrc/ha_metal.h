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
 * Uses MPS GEMM, LU decomposition, and triangular solve — no SVD needed.
 * Inputs/outputs are FP32. Operates entirely on GPU via unified memory.
 *
 * X: M x N, Y: M x N, T: N x N (pre-allocated).
 */
int ha_procrustes_metal(const TMatF *X, const TMatF *Y, TMatF *T,
                        bool isReflection, bool isScaling);

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

#endif // __APPLE__

#endif // HA_METAL_H
