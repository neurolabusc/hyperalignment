/*
 * ha_cuda.cu -- CUDA GPU backend for searchlight Procrustes alignment
 *
 * Algorithm per batch of searchlights:
 *   1. gather_kernel:             Extract columns from global X/Y into padded local buffers
 *   2. cublasSgemmStridedBatched: A_i = local_X_i^T @ local_Y_i  (column-major)
 *   3. gesvdaStridedBatched:      SVD: A_i = U_i * S_i * V_i^T
 *   4. cublasSgemmStridedBatched: T_i = U_i @ V_i^T              (column-major)
 *   5. scatter_add_kernel:        T_out[sl[r]*nv + sl[c]] += T_i[c,r] * w[r]
 *                                 (FP64 atomicAdd, native on CC >= 6.0)
 *
 * Data layout: column-major (Fortran order) matching Python F-contiguous input.
 * T_out uses row-major as expected by callers (standard C order).
 *
 * Padding: all matrices padded to max_sz for uniform batched operations.
 * Scatter-add only touches actual searchlight indices (0..sz-1 per searchlight).
 *
 * Limitations:
 *   - isReflection must be true (GPU path); reflection=false falls back in ha_searchlight.c
 *   - isScaling not supported on GPU
 *   - Requires CC >= 6.0 for native FP64 atomicAdd (RTX 4090 is CC 8.9)
 */

#include "ha_cuda.h"

#ifdef HA_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

static cublasHandle_t     g_cublas    = NULL;
static cusolverDnHandle_t g_cusolver  = NULL;
static bool               g_cuda_ready = false;

// ---------------------------------------------------------------------------
// Init / cleanup / available
// ---------------------------------------------------------------------------

int ha_cuda_init(void) {
    if (g_cuda_ready) return kHaSuccess;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        fprintf(stderr, "ha_cuda_init: no CUDA device\n");
        return kHaErrorInternal;
    }
    if (cudaSetDevice(0) != cudaSuccess) return kHaErrorInternal;
    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return kHaErrorInternal;
    if (cusolverDnCreate(&g_cusolver) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(g_cublas); g_cublas = NULL;
        return kHaErrorInternal;
    }
    g_cuda_ready = true;
    return kHaSuccess;
}

void ha_cuda_cleanup(void) {
    if (g_cusolver) { cusolverDnDestroy(g_cusolver); g_cusolver  = NULL; }
    if (g_cublas)   { cublasDestroy(g_cublas);       g_cublas    = NULL; }
    g_cuda_ready = false;
}

bool ha_cuda_available(void) { return g_cuda_ready; }

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

/*
 * gather_kernel
 * Extracts searchlight columns from global X_gpu/Y_gpu into local buffers.
 *
 * X_gpu, Y_gpu: column-major float32 (nt x nv). Column v at offset v*nt.
 *
 * local_X_batch, local_Y_batch: column-major float32 (nt x max_sz per searchlight).
 *   Searchlight batch_i at offset batch_i * nt * max_sz.
 *   Columns j >= actual_sz are zero-padded.
 *
 * Grid: dim3(batch_n, max_sz)   -- (searchlight, local_col) per block
 * Block: blockDim.x threads     -- partition nt timepoints
 */
__global__ void gather_kernel(
    const float * __restrict__ X_gpu,
    const float * __restrict__ Y_gpu,
    int nt, int,
    const int * __restrict__ sl_indices,
    const int * __restrict__ sl_offsets,
    int batch_start, int max_sz,
    float *local_X_batch, float *local_Y_batch)
{
    int bi  = (int)blockIdx.x;
    int lj  = (int)blockIdx.y;
    int si  = batch_start + bi;
    int off = sl_offsets[si];
    int sz  = sl_offsets[si + 1] - off;

    float *cx = local_X_batch + (long long)bi * nt * max_sz + (long long)lj * nt;
    float *cy = local_Y_batch + (long long)bi * nt * max_sz + (long long)lj * nt;

    if (lj < sz) {
        int gv = sl_indices[off + lj];
        const float *sx = X_gpu + (long long)gv * nt;
        const float *sy = Y_gpu + (long long)gv * nt;
        for (int t = (int)threadIdx.x; t < nt; t += (int)blockDim.x) {
            cx[t] = sx[t];
            cy[t] = sy[t];
        }
    } else {
        for (int t = (int)threadIdx.x; t < nt; t += (int)blockDim.x) {
            cx[t] = 0.0f;
            cy[t] = 0.0f;
        }
    }
}

/*
 * scatter_add_kernel
 * Accumulates local T matrices (FP32 col-major) into T_out (FP64 row-major).
 *
 * T[local_row, local_col] = T_batch[bi * max_sz^2 + local_col * max_sz + local_row]
 * T_out[global_row, global_col] = T_out[global_row * nv + global_col]
 *
 * Grid: dim3(batch_n, max_sz)   -- (searchlight, local_row) per block
 * Block: blockDim.x threads     -- partition local_col
 */
__global__ void scatter_add_kernel(
    const float * __restrict__ T_batch,
    double *T_out,
    const int * __restrict__ sl_indices,
    const int * __restrict__ sl_offsets,
    const double * __restrict__ weights,
    int batch_start, int max_sz, int nv)
{
    int bi   = (int)blockIdx.x;
    int lrow = (int)blockIdx.y;
    int si   = batch_start + bi;
    int off  = sl_offsets[si];
    int sz   = sl_offsets[si + 1] - off;
    if (lrow >= sz) return;

    int    grow = sl_indices[off + lrow];
    double wr   = weights[off + lrow];
    const float *T = T_batch + (long long)bi * max_sz * max_sz;

    for (int lc = (int)threadIdx.x; lc < sz; lc += (int)blockDim.x) {
        int    gcol = sl_indices[off + lc];
        double val  = (double)T[lc * max_sz + lrow] * wr;
        atomicAdd(&T_out[(long long)grow * nv + gcol], val);
    }
}

/*
 * d2f_kernel
 * Convert FP64 array to FP32. Grid: ceil(n/256), Block: 256.
 */
__global__ void d2f_kernel(const double * __restrict__ src, float *dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

// ---------------------------------------------------------------------------
// CPU weight computation
// ---------------------------------------------------------------------------

static int compute_weights_cpu(
    const int32_t *sl_indices, const int32_t *sl_offsets,
    const double *sl_dists, int32_t count, int32_t nv,
    double radius, double *out)
{
    double *wsum = (double *)calloc((size_t)nv, sizeof(double));
    if (!wsum) return kHaErrorAlloc;
    int32_t total = sl_offsets[count];
    if (!sl_dists) {
        for (int32_t k = 0; k < total; k++) wsum[sl_indices[k]] += 1.0;
        for (int32_t s = 0; s < count; s++) {
            int32_t off = sl_offsets[s], sz = sl_offsets[s + 1] - off;
            for (int32_t i = 0; i < sz; i++)
                out[off + i] = 1.0 / wsum[sl_indices[off + i]];
        }
    } else {
        for (int32_t k = 0; k < total; k++)
            wsum[sl_indices[k]] += (radius - sl_dists[k]) / radius;
        for (int32_t s = 0; s < count; s++) {
            int32_t off = sl_offsets[s], sz = sl_offsets[s + 1] - off;
            for (int32_t i = 0; i < sz; i++) {
                double w = (radius - sl_dists[off + i]) / radius;
                out[off + i] = w / wsum[sl_indices[off + i]];
            }
        }
    }
    free(wsum);
    return kHaSuccess;
}

// ---------------------------------------------------------------------------
// Batch size (searchlights per GPU command buffer)
// 256 * 1818 * 170 * 4 * 2 ≈ 630 MB for local_X + local_Y at typical sizes.
// ---------------------------------------------------------------------------
static const int CUDA_BATCH = 256;

// ---------------------------------------------------------------------------
// Inner batch processing (extracted to avoid goto-over-declaration in C++)
// ---------------------------------------------------------------------------

static int run_batches(
    cublasHandle_t cublas, cusolverDnHandle_t cusolver,
    float *d_X_f32, float *d_Y_f32,
    int32_t nt, int32_t nv, int32_t count, int32_t max_sz,
    const int *d_sl_indices, const int *d_sl_offsets, const double *d_weights,
    double *d_T_out)
{
    int32_t   batch_sz  = CUDA_BATCH;
    long long lX_elems  = (long long)batch_sz * nt * max_sz;
    long long A_elems   = (long long)batch_sz * max_sz * max_sz;
    long long S_elems   = (long long)batch_sz * max_sz;

    // All device pointers NULL-initialised for safe cudaFree on early exit
    float  *d_lX     = NULL;
    float  *d_lY     = NULL;
    float  *d_A      = NULL;
    float  *d_U      = NULL;
    float  *d_V      = NULL;
    float  *d_S      = NULL;
    float  *d_Tf32   = NULL;
    float  *d_work   = NULL;
    int    *d_info   = NULL;
    double *h_R_nrmF = NULL;
    int     lwork    = 0;
    int     error    = kHaSuccess;

    // cuBLAS scalars (declared before any potential early-return)
    const float alpha_f = 1.0f;
    const float beta_f  = 0.0f;

    // ---- Allocate batch buffers ----
    bool ok = true;
    ok = ok && (cudaMalloc(&d_lX,   lX_elems * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_lY,   lX_elems * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_A,    A_elems  * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_U,    A_elems  * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_V,    A_elems  * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_S,    S_elems  * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_Tf32, A_elems  * sizeof(float))  == cudaSuccess);
    ok = ok && (cudaMalloc(&d_info, batch_sz * sizeof(int))    == cudaSuccess);
    ok = ok && ((h_R_nrmF = (double *)malloc((size_t)batch_sz * sizeof(double))) != NULL);

    if (!ok) {
        fprintf(stderr, "ha_cuda: batch buffer allocation failed\n");
        error = kHaErrorAlloc;
    } else {
        // ---- Query SVD workspace ----
        cusolverStatus_t csr = cusolverDnSgesvdaStridedBatched_bufferSize(
            cusolver, CUSOLVER_EIG_MODE_VECTOR,
            max_sz, max_sz, max_sz,
            d_A, max_sz, (long long)max_sz * max_sz,
            d_S, (long long)max_sz,
            d_U, max_sz, (long long)max_sz * max_sz,
            d_V, max_sz, (long long)max_sz * max_sz,
            &lwork, batch_sz);
        if (csr != CUSOLVER_STATUS_SUCCESS) {
            fprintf(stderr, "ha_cuda: SVD workspace query failed (%d)\n", (int)csr);
            error = kHaErrorInternal;
        } else if (cudaMalloc(&d_work, (size_t)lwork * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "ha_cuda: SVD workspace allocation failed (%d floats)\n", lwork);
            error = kHaErrorAlloc;
        } else {
            // ---- Main batch loop ----
            for (int32_t bs = 0; bs < count && error == kHaSuccess; bs += batch_sz) {
                int32_t bn = (bs + batch_sz <= count) ? batch_sz : (count - bs);

                // Step 1: Gather
                {
                    int  tpb = (nt < 256) ? nt : 256;
                    dim3 grid((unsigned)bn, (unsigned)max_sz);
                    gather_kernel<<<grid, tpb>>>(
                        d_X_f32, d_Y_f32, nt, nv,
                        d_sl_indices, d_sl_offsets,
                        bs, max_sz, d_lX, d_lY);
                    if (cudaGetLastError() != cudaSuccess) { error = kHaErrorInternal; break; }
                }

                // Step 2: A = local_X^T @ local_Y  (col-major, CUBLAS_OP_T x CUBLAS_OP_N)
                if (cublasSgemmStridedBatched(
                        cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        max_sz, max_sz, nt,
                        &alpha_f,
                        d_lX, nt, (long long)nt * max_sz,
                        d_lY, nt, (long long)nt * max_sz,
                        &beta_f,
                        d_A, max_sz, (long long)max_sz * max_sz,
                        bn) != CUBLAS_STATUS_SUCCESS) { error = kHaErrorInternal; break; }

                // Step 3: SVD  A = U * S * V^T
                if (cusolverDnSgesvdaStridedBatched(
                        cusolver, CUSOLVER_EIG_MODE_VECTOR,
                        max_sz, max_sz, max_sz,
                        d_A,    max_sz, (long long)max_sz * max_sz,
                        d_S,    (long long)max_sz,
                        d_U,    max_sz, (long long)max_sz * max_sz,
                        d_V,    max_sz, (long long)max_sz * max_sz,
                        d_work, lwork,
                        d_info, h_R_nrmF,
                        bn) != CUSOLVER_STATUS_SUCCESS) { error = kHaErrorInternal; break; }
                cudaDeviceSynchronize();

                // Step 4: T = U @ V^T  (col-major, CUBLAS_OP_N x CUBLAS_OP_T)
                if (cublasSgemmStridedBatched(
                        cublas,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        max_sz, max_sz, max_sz,
                        &alpha_f,
                        d_U,   max_sz, (long long)max_sz * max_sz,
                        d_V,   max_sz, (long long)max_sz * max_sz,
                        &beta_f,
                        d_Tf32, max_sz, (long long)max_sz * max_sz,
                        bn) != CUBLAS_STATUS_SUCCESS) { error = kHaErrorInternal; break; }

                // Step 5: scatter-add T into T_out
                {
                    int  tpb  = (max_sz < 128) ? max_sz : 128;
                    dim3 grid((unsigned)bn, (unsigned)max_sz);
                    scatter_add_kernel<<<grid, tpb>>>(
                        d_Tf32, d_T_out,
                        d_sl_indices, d_sl_offsets, d_weights,
                        bs, max_sz, nv);
                    if (cudaGetLastError() != cudaSuccess) { error = kHaErrorInternal; break; }
                }
            } // batch loop

            cudaDeviceSynchronize();
        }
    }

    // Cleanup batch buffers (cudaFree(NULL) is a no-op)
    cudaFree(d_work);   free(h_R_nrmF);
    cudaFree(d_info);   cudaFree(d_Tf32);
    cudaFree(d_S);      cudaFree(d_V);
    cudaFree(d_U);      cudaFree(d_A);
    cudaFree(d_lY);     cudaFree(d_lX);
    return error;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

int ha_searchlight_procrustes_cuda(
    const double *X_data, const double *Y_data,
    int32_t nt, int32_t nv,
    const int32_t *sl_indices,
    const int32_t *sl_offsets,
    const double  *sl_dists,
    int32_t count,
    double radius,
    double *T_out,
    bool /*isReflection*/, bool /*isScaling*/,
    bool  /*col_major*/)
{
    if (!g_cuda_ready) return kHaErrorInternal;

    // Find max searchlight size
    int32_t max_sz = 0;
    for (int32_t s = 0; s < count; s++) {
        int32_t sz = sl_offsets[s + 1] - sl_offsets[s];
        if (sz > max_sz) max_sz = sz;
    }
    if (max_sz <= 0) return kHaErrorArg;

    int32_t total_nnz = sl_offsets[count];
    long long nv_nt   = (long long)nv * nt;

    // CPU weights
    double *h_weights = (double *)malloc((size_t)total_nnz * sizeof(double));
    if (!h_weights) return kHaErrorAlloc;
    {
        int rc = compute_weights_cpu(sl_indices, sl_offsets, sl_dists,
                                     count, nv, radius, h_weights);
        if (rc != kHaSuccess) { free(h_weights); return rc; }
    }

    // Device buffers (NULL-init for safe cudaFree on early exit)
    float  *d_X_f32      = NULL;
    float  *d_Y_f32      = NULL;
    double *d_T_out      = NULL;
    double *d_weights    = NULL;
    int    *d_sl_indices = NULL;
    int    *d_sl_offsets = NULL;
    int     error        = kHaSuccess;

    // Allocate persistent device buffers
    bool ok = true;
    ok = ok && (cudaMalloc(&d_X_f32,      nv_nt * sizeof(float))                       == cudaSuccess);
    ok = ok && (cudaMalloc(&d_Y_f32,      nv_nt * sizeof(float))                       == cudaSuccess);
    ok = ok && (cudaMalloc(&d_T_out,      (long long)nv * nv * sizeof(double))         == cudaSuccess);
    ok = ok && (cudaMalloc(&d_weights,    (size_t)total_nnz * sizeof(double))          == cudaSuccess);
    ok = ok && (cudaMalloc(&d_sl_indices, (size_t)total_nnz * sizeof(int))             == cudaSuccess);
    ok = ok && (cudaMalloc(&d_sl_offsets, (size_t)(count + 1) * sizeof(int))           == cudaSuccess);

    if (!ok) {
        fprintf(stderr, "ha_cuda: persistent GPU allocation failed\n");
        error = kHaErrorAlloc;
        goto done;
    }

    // Upload X, Y as FP64 then convert to FP32 on GPU
    {
        double *d_X_f64 = NULL;
        double *d_Y_f64 = NULL;
        bool stage_ok = true;
        stage_ok = stage_ok && (cudaMalloc(&d_X_f64, nv_nt * sizeof(double)) == cudaSuccess);
        stage_ok = stage_ok && (cudaMalloc(&d_Y_f64, nv_nt * sizeof(double)) == cudaSuccess);
        if (!stage_ok) {
            cudaFree(d_X_f64); cudaFree(d_Y_f64);
            error = kHaErrorAlloc;
            goto done;
        }
        bool upload_ok = true;
        upload_ok = upload_ok && (cudaMemcpy(d_X_f64, X_data, nv_nt * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
        upload_ok = upload_ok && (cudaMemcpy(d_Y_f64, Y_data, nv_nt * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
        if (!upload_ok) {
            cudaFree(d_X_f64); cudaFree(d_Y_f64);
            error = kHaErrorInternal;
            goto done;
        }
        int nb = (int)((nv_nt + 255) / 256);
        d2f_kernel<<<nb, 256>>>(d_X_f64, d_X_f32, nv_nt);
        d2f_kernel<<<nb, 256>>>(d_Y_f64, d_Y_f32, nv_nt);
        cudaFree(d_X_f64);
        cudaFree(d_Y_f64);
    }

    {
        bool copy_ok = true;
        copy_ok = copy_ok && (cudaMemcpy(d_weights,    h_weights,  (size_t)total_nnz * sizeof(double),    cudaMemcpyHostToDevice) == cudaSuccess);
        copy_ok = copy_ok && (cudaMemcpy(d_sl_indices, sl_indices, (size_t)total_nnz * sizeof(int32_t),   cudaMemcpyHostToDevice) == cudaSuccess);
        copy_ok = copy_ok && (cudaMemcpy(d_sl_offsets, sl_offsets, (size_t)(count + 1) * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
        copy_ok = copy_ok && (cudaMemset(d_T_out, 0, (long long)nv * nv * sizeof(double))                                         == cudaSuccess);
        if (!copy_ok) { error = kHaErrorInternal; goto done; }
    }

    // Run batched GPU processing
    error = run_batches(
        g_cublas, g_cusolver,
        d_X_f32, d_Y_f32, nt, nv, count, max_sz,
        d_sl_indices, d_sl_offsets, d_weights,
        d_T_out);

    // Download result
    if (error == kHaSuccess &&
        cudaMemcpy(T_out, d_T_out, (long long)nv * nv * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        error = kHaErrorInternal;

    done:
    cudaFree(d_sl_offsets); cudaFree(d_sl_indices);
    cudaFree(d_weights);    cudaFree(d_T_out);
    cudaFree(d_Y_f32);      cudaFree(d_X_f32);
    free(h_weights);
    return error;
}

#endif /* HA_CUDA_ENABLED */
