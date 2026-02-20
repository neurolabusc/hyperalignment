"""
hyperalignment_c.py -- Python ctypes wrapper for the C hyperalignment library.

Exposes searchlight_procrustes() with the same interface as the benchmark,
but dispatches to the C library for computation.

Usage:
    import hyperalignment_c as hac
    W = hac.searchlight_procrustes(X, Y, sls, dists, radius, backend='cpu64')
"""

import ctypes
import os
import sys
import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------

_libdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

if sys.platform == "darwin":
    _libname = "libhyperalignment.dylib"
else:
    _libname = "libhyperalignment.so"

_libpath = os.path.join(_libdir, _libname)
if not os.path.exists(_libpath):
    raise FileNotFoundError(
        f"Shared library not found at {_libpath}. "
        f"Run 'make' in {_libdir} first."
    )

_lib = ctypes.CDLL(_libpath)


# ---------------------------------------------------------------------------
# ctypes struct definitions matching C types in ha_common.h
# ---------------------------------------------------------------------------

class TMat(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
    ]


class TSearchlights(ctypes.Structure):
    _fields_ = [
        ("indices", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("dists", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ("sizes", ctypes.POINTER(ctypes.c_int32)),
        ("count", ctypes.c_int32),
        ("radius", ctypes.c_double),
    ]


class TSparseCSC(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("indices", ctypes.POINTER(ctypes.c_int32)),
        ("indptr", ctypes.POINTER(ctypes.c_int32)),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("nnz", ctypes.c_int64),
    ]


# ---------------------------------------------------------------------------
# C function signatures
# ---------------------------------------------------------------------------

# ha_searchlight_weights(const TSearchlights *sls, double **weights_out)
_lib.ha_searchlight_weights.restype = ctypes.c_int
_lib.ha_searchlight_weights.argtypes = [
    ctypes.POINTER(TSearchlights),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
]

# ha_sparse_init(const TSearchlights *sls, int32_t nv)
_lib.ha_sparse_init.restype = ctypes.POINTER(TSparseCSC)
_lib.ha_sparse_init.argtypes = [
    ctypes.POINTER(TSearchlights),
    ctypes.c_int32,
]

# libc free() for manual cleanup (ha_sparse_free is static inline, not exported)
_libc = ctypes.CDLL(None)
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]

# ha_searchlight_procrustes(...)
_lib.ha_searchlight_procrustes.restype = ctypes.c_int
_lib.ha_searchlight_procrustes.argtypes = [
    ctypes.POINTER(TMat),                              # X
    ctypes.POINTER(TMat),                              # Y
    ctypes.POINTER(TSearchlights),                     # sls_X
    ctypes.POINTER(TSearchlights),                     # sls_Y (can be NULL)
    ctypes.POINTER(TSparseCSC),                        # mat
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),   # weights
    ctypes.c_bool,                                     # isReflection
    ctypes.c_bool,                                     # isScaling
    ctypes.c_int,                                      # backend (THaBackend)
]

# Metal init/cleanup
_lib.ha_metal_init.restype = ctypes.c_int
_lib.ha_metal_init.argtypes = []

_lib.ha_metal_cleanup.restype = None
_lib.ha_metal_cleanup.argtypes = []

_lib.ha_metal_available.restype = ctypes.c_bool
_lib.ha_metal_available.argtypes = []

# Thread control
_lib.ha_set_num_threads.restype = None
_lib.ha_set_num_threads.argtypes = [ctypes.c_int]

_lib.ha_get_num_threads.restype = ctypes.c_int
_lib.ha_get_num_threads.argtypes = []


# ---------------------------------------------------------------------------
# Backend enum values (matches THaBackend in ha_common.h)
# ---------------------------------------------------------------------------

_BACKENDS = {
    "cpu64": 0,
    "c64": 0,
    "cpu32": 1,
    "c32": 1,
    "metal": 2,
}


# ---------------------------------------------------------------------------
# Helper: build C structs from numpy arrays
# ---------------------------------------------------------------------------

def _make_mat(arr):
    """Wrap a contiguous float64 numpy array as a TMat (zero-copy)."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    mat = TMat()
    mat.data = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mat.rows = ctypes.c_int32(arr.shape[0])
    mat.cols = ctypes.c_int32(arr.shape[1])
    return mat, arr  # return arr to prevent GC


def _make_searchlights(sls, dists, radius):
    """
    Build a TSearchlights struct from Python lists.

    sls: list of numpy int arrays (vertex indices per searchlight)
    dists: list of numpy float arrays (distances per searchlight)
    radius: float

    Returns (TSearchlights, keepalive_list) where keepalive_list must be kept
    alive to prevent GC of the underlying buffers.
    """
    count = len(sls)
    keepalive = []

    # indices: array of int32* pointers
    IndexPtrArray = ctypes.POINTER(ctypes.c_int32) * count
    idx_ptrs = IndexPtrArray()
    sizes_arr = (ctypes.c_int32 * count)()
    for i, sl in enumerate(sls):
        sl_i32 = np.ascontiguousarray(sl, dtype=np.int32)
        keepalive.append(sl_i32)
        idx_ptrs[i] = sl_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        sizes_arr[i] = ctypes.c_int32(len(sl_i32))

    # dists: array of double* pointers (or NULL)
    DistPtrArray = ctypes.POINTER(ctypes.c_double) * count
    dist_ptrs = DistPtrArray()
    for i, d in enumerate(dists):
        d_f64 = np.ascontiguousarray(d, dtype=np.float64)
        keepalive.append(d_f64)
        dist_ptrs[i] = d_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    tsls = TSearchlights()
    tsls.indices = ctypes.cast(idx_ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
    tsls.dists = ctypes.cast(dist_ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    tsls.sizes = ctypes.cast(sizes_arr, ctypes.POINTER(ctypes.c_int32))
    tsls.count = ctypes.c_int32(count)
    tsls.radius = ctypes.c_double(radius)

    # Keep the pointer arrays alive too
    keepalive.extend([idx_ptrs, dist_ptrs, sizes_arr])
    return tsls, keepalive


def _sparse_to_scipy(mat_ptr):
    """
    Convert a TSparseCSC* to a scipy.sparse.csc_matrix.
    Copies the data (since C memory will be freed).
    """
    mat = mat_ptr.contents
    nnz = mat.nnz
    nrows = mat.rows
    ncols = mat.cols

    # Copy C arrays into numpy
    data = np.ctypeslib.as_array(mat.data, shape=(nnz,)).copy()
    indices = np.ctypeslib.as_array(mat.indices, shape=(nnz,)).copy()
    indptr = np.ctypeslib.as_array(mat.indptr, shape=(ncols + 1,)).copy()

    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(nrows, ncols))


def _free_sparse(mat_ptr):
    """Free a TSparseCSC* and its member arrays (mirrors ha_sparse_free)."""
    if mat_ptr:
        mat = mat_ptr.contents
        if mat.data:
            _libc.free(ctypes.cast(mat.data, ctypes.c_void_p))
        if mat.indices:
            _libc.free(ctypes.cast(mat.indices, ctypes.c_void_p))
        if mat.indptr:
            _libc.free(ctypes.cast(mat.indptr, ctypes.c_void_p))
        _libc.free(ctypes.cast(mat_ptr, ctypes.c_void_p))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def searchlight_procrustes(X, Y, sls, dists, radius, backend="cpu64",
                           reflection=True, scaling=False, n_jobs=1):
    """
    Searchlight Procrustes alignment using the C hyperalignment library.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix to align.
    Y : ndarray of shape (n_samples, n_features)
        Target data matrix.
    sls : list of ndarray
        Searchlight vertex indices.
    dists : list of ndarray
        Distances from searchlight centers.
    radius : float
        Searchlight radius.
    backend : str
        One of 'cpu64' (default), 'cpu32', 'metal'.
    reflection : bool
        Allow reflection in Procrustes (default True).
    scaling : bool
        Allow scaling in Procrustes (default False).
    n_jobs : int
        Number of OpenMP threads. Default 1 (single-threaded).
        Requires the library to be compiled with OPENMP=1.

    Returns
    -------
    W : scipy.sparse.csc_matrix
        Transformation matrix of shape (n_features, n_features).
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_BACKENDS.keys())}")

    backend_enum = _BACKENDS[backend]

    # Set thread count
    if n_jobs > 0:
        _lib.ha_set_num_threads(n_jobs)

    # Build C structs
    X_c, X_keep = _make_mat(X)
    Y_c, Y_keep = _make_mat(Y)
    sls_c, sls_keep = _make_searchlights(sls, dists, radius)
    n_sls = len(sls)

    # Compute weights
    WeightPtrArray = ctypes.POINTER(ctypes.c_double) * n_sls
    weights_ptrs = WeightPtrArray()
    rc = _lib.ha_searchlight_weights(ctypes.byref(sls_c), weights_ptrs)
    if rc != 0:
        raise RuntimeError(f"ha_searchlight_weights failed with error code {rc}")

    # Initialize sparse matrix
    mat_ptr = _lib.ha_sparse_init(ctypes.byref(sls_c), ctypes.c_int32(0))
    if not mat_ptr:
        _free_weights(weights_ptrs, n_sls)
        raise RuntimeError("ha_sparse_init failed (returned NULL)")

    # Metal init if needed
    metal_inited = False
    if backend_enum == 2:  # kHaBackendMetal
        rc = _lib.ha_metal_init()
        if rc == 0:
            metal_inited = True

    # Run searchlight Procrustes
    rc = _lib.ha_searchlight_procrustes(
        ctypes.byref(X_c),
        ctypes.byref(Y_c),
        ctypes.byref(sls_c),
        None,  # sls_Y = NULL (same as sls_X)
        mat_ptr,
        weights_ptrs,
        ctypes.c_bool(reflection),
        ctypes.c_bool(scaling),
        ctypes.c_int(backend_enum),
    )
    if rc != 0:
        _free_weights(weights_ptrs, n_sls)
        _free_sparse(mat_ptr)
        if metal_inited:
            _lib.ha_metal_cleanup()
        raise RuntimeError(f"ha_searchlight_procrustes failed with error code {rc}")

    # Convert result to scipy sparse
    result = _sparse_to_scipy(mat_ptr)

    # Cleanup C memory
    _free_weights(weights_ptrs, n_sls)
    _free_sparse(mat_ptr)
    if metal_inited:
        _lib.ha_metal_cleanup()

    return result


def _free_weights(weights_ptrs, n_sls):
    """Free each weight array allocated by ha_searchlight_weights."""
    for i in range(n_sls):
        if weights_ptrs[i]:
            _libc.free(ctypes.cast(weights_ptrs[i], ctypes.c_void_p))
