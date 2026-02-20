/*
 * ha_metal.m -- Metal GPU backend implementation
 *
 * Uses Metal Performance Shaders for GPU-accelerated Procrustes alignment.
 * GPU handles the GEMM (X^T @ Y), CPU handles the Newton iteration.
 *
 * Batch API: encodes multiple GEMMs into a single command buffer and
 * submits asynchronously, enabling pipelining with CPU Newton iteration.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "ha_metal.h"
#include "ha_procrustes.h"

// Singleton Metal context
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;

int ha_metal_init(void) {
	if (g_device) return kHaSuccess;
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

// Create an MPSMatrix wrapping a float buffer (copies data into GPU-visible buffer)
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

// Create an MPSMatrix with its own uninitialized buffer
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

// ---- Single-searchlight Procrustes (existing API) ----

int ha_procrustes_metal(const TMatF *X, const TMatF *Y, TMatF *T,
                        bool isReflection, bool isScaling) {
	if (!g_device) return kHaErrorInternal;

	int32_t M = X->rows;
	int32_t N = X->cols;

	// GPU GEMM: A = X^T @ Y
	@autoreleasepool {
		MPSMatrix *mX = mps_matrix_from_float(X->data, M, N);
		MPSMatrix *mY = mps_matrix_from_float(Y->data, M, N);
		MPSMatrix *mA = mps_matrix_alloc(N, N);
		if (!mX || !mY || !mA) return kHaErrorAlloc;

		MPSMatrixMultiplication *kernel =
			[[MPSMatrixMultiplication alloc] initWithDevice:g_device
			                                 transposeLeft:true
			                                transposeRight:false
			                                    resultRows:N
			                                 resultColumns:N
			                               interiorColumns:M
			                                         alpha:1.0f
			                                          beta:0.0f];
		id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
		[kernel encodeToCommandBuffer:cmdBuf leftMatrix:mX rightMatrix:mY resultMatrix:mA];
		[cmdBuf commit];
		[cmdBuf waitUntilCompleted];
		if (cmdBuf.error) return kHaErrorInternal;

		// Read A and run Newton iteration on CPU
		float *A = (float *)malloc((size_t)N * N * sizeof(float));
		if (!A) return kHaErrorAlloc;
		mps_matrix_read(mA, A, N, N);

		int rc = ha_polar_newton(A, T->data, N, isReflection, isScaling, X->data, M);
		free(A);
		if (rc != kHaSuccess)
			return ha_procrustes_f32(X, Y, T, isReflection, isScaling);
	}

	return kHaSuccess;
}

// ---- Batched GEMM API ----

struct HaMetalBatch {
	void *cmdBuf;       // __bridge_retained id<MTLCommandBuffer>
	void *results;      // __bridge_retained NSArray<MPSMatrix *>
	int32_t count;
	int32_t *sizes;
};

HaMetalBatch *ha_metal_batch_submit(int32_t count, int32_t nt,
                                     const int32_t *sizes,
                                     float *const *X_data,
                                     float *const *Y_data) {
	if (!g_device || count <= 0) return NULL;

	HaMetalBatch *batch = (HaMetalBatch *)calloc(1, sizeof(HaMetalBatch));
	if (!batch) return NULL;
	batch->count = count;
	batch->sizes = (int32_t *)malloc((size_t)count * sizeof(int32_t));
	if (!batch->sizes) { free(batch); return NULL; }
	memcpy(batch->sizes, sizes, (size_t)count * sizeof(int32_t));

	@autoreleasepool {
		NSMutableArray<MPSMatrix *> *results = [NSMutableArray arrayWithCapacity:count];
		id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
		if (!cmdBuf) { free(batch->sizes); free(batch); return NULL; }

		for (int32_t i = 0; i < count; i++) {
			int32_t N = sizes[i];
			MPSMatrix *mX = mps_matrix_from_float(X_data[i], nt, N);
			MPSMatrix *mY = mps_matrix_from_float(Y_data[i], nt, N);
			MPSMatrix *mA = mps_matrix_alloc(N, N);
			if (!mX || !mY || !mA) {
				free(batch->sizes); free(batch);
				return NULL;
			}
			[results addObject:mA];

			MPSMatrixMultiplication *kernel =
				[[MPSMatrixMultiplication alloc] initWithDevice:g_device
				                                 transposeLeft:true
				                                transposeRight:false
				                                    resultRows:N
				                                 resultColumns:N
				                               interiorColumns:nt
				                                         alpha:1.0f
				                                          beta:0.0f];
			[kernel encodeToCommandBuffer:cmdBuf leftMatrix:mX rightMatrix:mY resultMatrix:mA];
		}

		[cmdBuf commit];  // non-blocking submit

		// Retain ObjC objects for lifetime beyond autoreleasepool
		batch->cmdBuf = (__bridge_retained void *)cmdBuf;
		batch->results = (__bridge_retained void *)results;
	}

	return batch;
}

int ha_metal_batch_wait(HaMetalBatch *batch, float **A_out) {
	if (!batch || !batch->cmdBuf) return kHaErrorInternal;

	id<MTLCommandBuffer> cmdBuf = (__bridge id<MTLCommandBuffer>)batch->cmdBuf;
	[cmdBuf waitUntilCompleted];
	if (cmdBuf.error) return kHaErrorInternal;

	NSArray<MPSMatrix *> *results = (__bridge NSArray<MPSMatrix *> *)batch->results;
	for (int32_t i = 0; i < batch->count; i++) {
		int32_t N = batch->sizes[i];
		memcpy(A_out[i], results[i].data.contents, (size_t)N * N * sizeof(float));
	}

	return kHaSuccess;
}

void ha_metal_batch_free(HaMetalBatch *batch) {
	if (!batch) return;
	@autoreleasepool {
		if (batch->cmdBuf)
			(void)(__bridge_transfer id)batch->cmdBuf;
		if (batch->results)
			(void)(__bridge_transfer id)batch->results;
	}
	free(batch->sizes);
	free(batch);
}
