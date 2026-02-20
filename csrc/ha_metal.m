/*
 * ha_metal.m -- Metal GPU backend implementation
 *
 * Uses Metal Performance Shaders for GPU-accelerated Procrustes alignment
 * via polar decomposition (Newton iteration). All GPU operations are FP32.
 *
 * Polar decomposition: A = U_p * H where U_p is the closest orthogonal
 * matrix to A. Newton iteration: X_{k+1} = (X_k + X_k^{-T}) / 2
 * converges to U_p in ~6-10 iterations.
 *
 * Each iteration uses:
 *   1. LU decomposition (MPSMatrixDecompositionLU)
 *   2. Triangular solves for inverse (MPSMatrixSolveTriangular)
 *   3. Transpose + average (CPU, small matrix)
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "ha_metal.h"
#include "ha_procrustes.h"

// Singleton Metal context
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

int ha_metal_init(void) {
	if (g_device) return kHaSuccess;  // already initialized
	g_device = MTLCreateSystemDefaultDevice();
	if (!g_device) return kHaErrorInternal;
	g_queue = [g_device newCommandQueue];
	if (!g_queue) {
		g_device = nil;
		return kHaErrorInternal;
	}
	return kHaSuccess;
}

void ha_metal_cleanup(void) {
	g_queue = nil;
	g_device = nil;
}

bool ha_metal_available(void) {
	return g_device != nil;
}

// Create an MPSMatrix wrapping a float buffer (shared memory — zero-copy on Apple Silicon)
static MPSMatrix *mps_matrix_from_float(float *data, int32_t rows, int32_t cols) {
	NSUInteger rowBytes = (NSUInteger)cols * sizeof(float);
	NSUInteger totalBytes = (NSUInteger)rows * rowBytes;
	id<MTLBuffer> buffer = [g_device newBufferWithBytes:data
	                                            length:totalBytes
	                                           options:MTLResourceStorageModeShared];
	if (!buffer) return nil;
	MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
	                                                                 columns:cols
	                                                                rowBytes:rowBytes
	                                                                dataType:MPSDataTypeFloat32];
	return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
}

// Create an MPSMatrix with its own buffer (for intermediate results)
static MPSMatrix *mps_matrix_alloc(int32_t rows, int32_t cols) {
	NSUInteger rowBytes = (NSUInteger)cols * sizeof(float);
	NSUInteger totalBytes = (NSUInteger)rows * rowBytes;
	id<MTLBuffer> buffer = [g_device newBufferWithLength:totalBytes
	                                            options:MTLResourceStorageModeShared];
	if (!buffer) return nil;
	MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
	                                                                 columns:cols
	                                                                rowBytes:rowBytes
	                                                                dataType:MPSDataTypeFloat32];
	return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
}

// Read float data from an MPSMatrix buffer
static void mps_matrix_read(const MPSMatrix *mat, float *data, int32_t rows, int32_t cols) {
	memcpy(data, mat.data.contents, (size_t)rows * cols * sizeof(float));
}

// GPU GEMM: C = alpha * op(A) * op(B) + beta * C
static int mps_gemm(MPSMatrix *A, MPSMatrix *B, MPSMatrix *C,
                     bool transA, bool transB,
                     float alpha, float beta) {
	MPSMatrixMultiplication *kernel =
		[[MPSMatrixMultiplication alloc] initWithDevice:g_device
		                                 transposeLeft:transA
		                                transposeRight:transB
		                                    resultRows:C.rows
		                                 resultColumns:C.columns
		                               interiorColumns:(transA ? A.rows : A.columns)
		                                         alpha:alpha
		                                          beta:beta];
	id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
	[kernel encodeToCommandBuffer:cmdBuf leftMatrix:A rightMatrix:B resultMatrix:C];
	[cmdBuf commit];
	[cmdBuf waitUntilCompleted];
	return (cmdBuf.error == nil) ? kHaSuccess : kHaErrorInternal;
}

// Transpose an N x N matrix in-place (CPU, small matrix)
static void transpose_inplace(float *data, int32_t N) {
	for (int32_t i = 0; i < N; i++)
		for (int32_t j = i + 1; j < N; j++) {
			float tmp = data[i * N + j];
			data[i * N + j] = data[j * N + i];
			data[j * N + i] = tmp;
		}
}

// Compute matrix inverse via CPU LAPACK (sgetrf + sgetri).
// MPS LU + triangular solves have complex pivot handling; for the small
// matrices we deal with (~200x200), CPU LAPACK inverse is fast and correct.
// Input: A data is copied, Output: Ainv data is filled with A^{-1}.
static int matrix_inverse_cpu(const float *A_data, float *Ainv_data, int32_t N) {
	memcpy(Ainv_data, A_data, (size_t)N * N * sizeof(float));

	ha_lapack_int lN = N, info = 0;
	ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
	if (!ipiv) return kHaErrorAlloc;

	// LU factorization (column-major, but det(A)=det(A^T) and we just need inverse)
	// For inverse, row-major vs col-major gives (A^T)^{-1} = (A^{-1})^T,
	// but since we immediately transpose in Newton iteration, this is fine:
	// We compute (A^T)^{-1} in row-major layout, then transpose to get A^{-1}.
	// Actually, to keep it simple, just compute in column-major style:
	// LAPACK sees our row-major A as A^T in column-major.
	// sgetrf gives P * A^T = L * U
	// sgetri gives (A^T)^{-1}
	// When read as row-major, this is (A^{-1})^T.
	// Since Newton iteration needs A^{-T}, this is exactly what we want!
	// So we skip the explicit transpose after inverse.

	sgetrf_(&lN, &lN, Ainv_data, &lN, ipiv, &info);
	if (info != 0) {
		free(ipiv);
		return kHaErrorInternal;  // singular
	}

	// Query optimal workspace
	float work_query;
	ha_lapack_int lwork = -1;
	sgetri_(&lN, Ainv_data, &lN, ipiv, &work_query, &lwork, &info);
	lwork = (ha_lapack_int)work_query;
	float *work = (float *)malloc((size_t)lwork * sizeof(float));
	if (!work) { free(ipiv); return kHaErrorAlloc; }

	sgetri_(&lN, Ainv_data, &lN, ipiv, work, &lwork, &info);
	free(work);
	free(ipiv);

	return (info == 0) ? kHaSuccess : kHaErrorInternal;
}

// Frobenius norm of difference: ||A - B||_F / ||A||_F
static float frobenius_reldiff(const float *a, const float *b, int32_t n) {
	float diff_sq = 0.0f, norm_sq = 0.0f;
	for (int32_t i = 0; i < n; i++) {
		float d = a[i] - b[i];
		diff_sq += d * d;
		norm_sq += a[i] * a[i];
	}
	return sqrtf(diff_sq) / (sqrtf(norm_sq) + 1e-30f);
}

int ha_procrustes_metal(const TMatF *X, const TMatF *Y, TMatF *T,
                        bool isReflection, bool isScaling) {
	if (!g_device) return kHaErrorInternal;

	int32_t M = X->rows;
	int32_t N = X->cols;

	// GPU matrices
	MPSMatrix *mX = mps_matrix_from_float(X->data, M, N);
	MPSMatrix *mY = mps_matrix_from_float(Y->data, M, N);
	MPSMatrix *mA = mps_matrix_alloc(N, N);
	if (!mX || !mY || !mA) return kHaErrorAlloc;

	// A = X^T @ Y (N x N)
	int rc = mps_gemm(mX, mY, mA, /*transA=*/true, /*transB=*/false, 1.0f, 0.0f);
	if (rc != kHaSuccess) return rc;

	// Polar decomposition via Newton iteration
	// X_0 = A, X_{k+1} = (X_k + X_k^{-T}) / 2
	float *Xk = (float *)malloc((size_t)N * N * sizeof(float));
	float *Xkinvt = (float *)malloc((size_t)N * N * sizeof(float));
	float *prev = (float *)malloc((size_t)N * N * sizeof(float));
	if (!Xk || !Xkinvt || !prev) {
		free(Xk); free(Xkinvt); free(prev);
		return kHaErrorAlloc;
	}

	// X_0 = A
	mps_matrix_read(mA, Xk, N, N);

	int32_t max_iter = 20;
	float tol = 1e-6f;
	for (int32_t iter = 0; iter < max_iter; iter++) {
		memcpy(prev, Xk, (size_t)N * N * sizeof(float));

		// Compute X_k^{-T} via CPU LAPACK
		// matrix_inverse_cpu returns A^{-1} in row-major; transpose to get A^{-T}
		rc = matrix_inverse_cpu(Xk, Xkinvt, N);
		if (rc == kHaSuccess)
			transpose_inplace(Xkinvt, N);
		if (rc != kHaSuccess) {
			// If matrix is singular, fall back to CPU FP32 SVD path
			free(Xk); free(Xkinvt); free(prev);
			return ha_procrustes_f32(X, Y, T, isReflection, isScaling);
		}

		// X_{k+1} = (X_k + X_k^{-T}) / 2
		for (int32_t i = 0; i < N * N; i++)
			Xk[i] = 0.5f * (Xk[i] + Xkinvt[i]);

		// Check convergence
		if (frobenius_reldiff(Xk, prev, N * N) < tol)
			break;
	}
	free(Xkinvt);
	free(prev);

	// T = U_p (the converged orthogonal factor)
	memcpy(T->data, Xk, (size_t)N * N * sizeof(float));
	free(Xk);

	// Handle reflection
	if (!isReflection) {
		// Compute det(T) via LU on CPU (small matrix, not worth GPU)
		float *T_tmp = (float *)malloc((size_t)N * N * sizeof(float));
		ha_lapack_int *ipiv = (ha_lapack_int *)malloc((size_t)N * sizeof(ha_lapack_int));
		if (!T_tmp || !ipiv) { free(T_tmp); free(ipiv); return kHaErrorAlloc; }
		memcpy(T_tmp, T->data, (size_t)N * N * sizeof(float));

		ha_lapack_int lN = N, info = 0;
		sgetrf_(&lN, &lN, T_tmp, &lN, ipiv, &info);

		float sign = 1.0f;
		for (int32_t i = 0; i < N; i++) {
			if (T_tmp[i * N + i] < 0.0f) sign = -sign;
			if (ipiv[i] != i + 1) sign = -sign;
		}
		free(T_tmp);
		free(ipiv);

		if (sign < 0.0f) {
			// Flip last column of T and negate to correct reflection
			// For polar decomposition, we need to correct the factor
			// T_corrected = T * diag(1,...,1,-1) * diag(1,...,1,-1) adjustment
			// Simplest: negate last column of T
			for (int32_t i = 0; i < N; i++)
				T->data[i * N + (N - 1)] *= -1.0f;
		}
	}

	// Handle scaling
	if (isScaling) {
		// H = U_p^T @ A, scale = trace(H) / (var(X) * M)
		MPSMatrix *mH = mps_matrix_alloc(N, N);
		MPSMatrix *mT = mps_matrix_from_float(T->data, N, N);
		if (!mH || !mT) return kHaErrorAlloc;

		mps_gemm(mT, mA, mH, /*transA=*/true, /*transB=*/false, 1.0f, 0.0f);

		float *H_data = (float *)mH.data.contents;
		float trace_H = 0.0f;
		for (int32_t i = 0; i < N; i++)
			trace_H += H_data[i * N + i];

		// Compute var(X) * M
		float var_sum = 0.0f;
		for (int32_t j = 0; j < N; j++) {
			float mean = 0.0f;
			for (int32_t i = 0; i < M; i++)
				mean += X->data[i * N + j];
			mean /= M;
			float var = 0.0f;
			for (int32_t i = 0; i < M; i++) {
				float diff = X->data[i * N + j] - mean;
				var += diff * diff;
			}
			var_sum += var / M;
		}

		float scale = trace_H / (var_sum * M);
		cblas_sscal(N * N, scale, T->data, 1);
	}

	return kHaSuccess;
}
