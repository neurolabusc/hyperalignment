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
# C function signatures
# ---------------------------------------------------------------------------

# ha_searchlight_procrustes_dense(...)
_lib.ha_searchlight_procrustes_dense.restype = ctypes.c_int
_lib.ha_searchlight_procrustes_dense.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # X_data
    ctypes.POINTER(ctypes.c_double),   # Y_data
    ctypes.c_int32,                    # nt
    ctypes.c_int32,                    # nv
    ctypes.POINTER(ctypes.c_int32),    # sl_indices
    ctypes.POINTER(ctypes.c_int32),    # sl_offsets
    ctypes.POINTER(ctypes.c_double),   # sl_dists (can be NULL)
    ctypes.c_int32,                    # count
    ctypes.c_double,                   # radius
    ctypes.POINTER(ctypes.c_double),   # T_out
    ctypes.c_bool,                     # isReflection
    ctypes.c_bool,                     # isScaling
    ctypes.c_int,                      # backend (THaBackend)
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
    T : ndarray of shape (n_features, n_features)
        Dense transformation matrix.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_BACKENDS.keys())}")

    backend_enum = _BACKENDS[backend]

    # Set thread count
    if n_jobs > 0:
        _lib.ha_set_num_threads(n_jobs)

    # Ensure contiguous float64
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    nt, nv = X.shape

    # Build flat arrays from searchlight lists (vectorized numpy ops)
    sizes = np.array([len(s) for s in sls], dtype=np.int32)
    offsets = np.zeros(len(sls) + 1, dtype=np.int32)
    np.cumsum(sizes, out=offsets[1:])
    all_indices = np.concatenate(sls).astype(np.int32)

    if dists is not None:
        all_dists = np.concatenate(dists).astype(np.float64)
        dists_ptr = all_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        all_dists = None
        dists_ptr = None

    # Allocate dense output (zero-initialized by numpy)
    T = np.zeros((nv, nv), dtype=np.float64)

    # Metal init if needed
    metal_inited = False
    if backend_enum == 2:  # kHaBackendMetal
        rc = _lib.ha_metal_init()
        if rc == 0:
            metal_inited = True

    # Single C call
    rc = _lib.ha_searchlight_procrustes_dense(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(nt),
        ctypes.c_int32(nv),
        all_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        dists_ptr,
        ctypes.c_int32(len(sls)),
        ctypes.c_double(radius),
        T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_bool(reflection),
        ctypes.c_bool(scaling),
        ctypes.c_int(backend_enum),
    )

    if metal_inited:
        _lib.ha_metal_cleanup()

    if rc != 0:
        raise RuntimeError(f"ha_searchlight_procrustes_dense failed with error code {rc}")

    return T
