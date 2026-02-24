#!/usr/bin/env python3
"""
benchmark_neuroboros.py

Usage:
    python benchmark_neuroboros.py /path/to/neuroboros_backup
    python benchmark_neuroboros.py /path/to/neuroboros_backup --backend c64 --n-jobs 10 --repeat 3

This script forces neuroboros to use the local data directory you provide,
runs the benchmark code, saves r_test0 to disk (NPZ + NPY), and prints elapsed time.
"""
import os
import sys
import time
import argparse
import traceback
import numpy as np
import datetime as dt

def find_forrest_root(base_dir):
    """
    Given a base dir that should contain 'forrest' and 'core' (like your neuroboros_backup),
    try to resolve the exact dataset root (e.g. base/forrest/20.2.7). Returns a path or None.
    """
    # If user passed a path that already looks like the exact dataset root, return it.
    if os.path.isdir(os.path.join(base_dir, "resampled")) and os.path.isdir(os.path.join(base_dir, "confounds")):
        return base_dir

    # If base contains 'forrest' subdirectory, look for versioned subdirs under it.
    forrest_dir = os.path.join(base_dir, "forrest")
    if os.path.isdir(forrest_dir):
        # if forrest contains version-like folders, pick the first one
        entries = [e for e in os.listdir(forrest_dir) if os.path.isdir(os.path.join(forrest_dir, e))]
        if len(entries) == 1:
            candidate = os.path.join(forrest_dir, entries[0])
            return candidate
        # if no subfolders or many, prefer a folder that looks like a version "20." or "v"
        for e in entries:
            if e.startswith("20") or e.startswith("v"):
                return os.path.join(forrest_dir, e)
        # fallback: return the forrest dir itself
        return forrest_dir

    # If base itself is something like .../forrest/20.2.7 (user passed deep path)
    if "forrest" in os.path.basename(base_dir).lower() or "forrest" in base_dir.lower():
        return base_dir

    return None

def main():
    p = argparse.ArgumentParser(description="Run neuroboros benchmark using a local copy of neuroboros data.")
    p.add_argument("data_dir", help="Path to directory that contains 'forrest' and 'core' (your neuroboros_backup folder).")
    p.add_argument("--outdir", default=None, help="Optional output directory (defaults to <data_dir>/results).")
    p.add_argument("--backend", choices=["python", "c64", "c32", "metal", "cuda"],
                   default="python",
                   help="Backend: python (default), c64 (C CPU FP64), c32 (C CPU FP32), "
                        "metal (C Metal GPU FP32, macOS only), cuda (C CUDA GPU FP32, NVIDIA)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Number of OpenMP threads for C backends (default: 1). Requires OPENMP=1 build.")
    p.add_argument("--repeat", type=int, default=1,
                   help="Number of times to repeat the alignment (default: 1). Reports median.")
    args = p.parse_args()

    base = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(base):
        print(f"ERROR: data_dir does not exist or is not a directory: {base}", file=sys.stderr)
        sys.exit(2)

    # Set environment variable before importing neuroboros so the package can pick it up
    # (many data packages respect such an env var; this is non-destructive and safe).
    os.environ.setdefault("NEUROBOROS_DATA", base)
    print(f"Using NEUROBOROS_DATA = {os.environ['NEUROBOROS_DATA']}")

    # Import after env var set
    try:
        import neuroboros as nb
        from scipy.stats import zscore
        if args.backend == "python":
            import hyperalignment as ha
        else:
            import hyperalignment_c as hac
            # Verify CUDA library was built with CUDA support
            if args.backend == "cuda":
                import ctypes
                if not hac._lib.ha_cuda_available():
                    # ha_cuda_init hasn't been called yet; init to check
                    rc = hac._lib.ha_cuda_init()
                    if rc != 0:
                        print("WARNING: CUDA backend unavailable. "
                              "Rebuild with 'make CUDA=1 OPENMP=1'.", file=sys.stderr)
    except Exception:
        print("Failed to import required packages. Traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    print(f"Backend: {args.backend}, n_jobs: {args.n_jobs}, repeat: {args.repeat}")

    # Instantiate the dataset
    dset = nb.Forrest()

    # If dataset object didn't set a root_dir, try to point it to local copy
    try:
        current_root = getattr(dset, "root_dir", None)
    except Exception:
        current_root = None

    if current_root:
        print(f"dset.root_dir already set to: {current_root!r}")
    else:
        guessed = find_forrest_root(base)
        if guessed:
            try:
                # assign root_dir attribute if present / writable
                setattr(dset, "root_dir", guessed)
                print(f"Set dset.root_dir -> {guessed}")
            except Exception:
                print("Could not set dset.root_dir attribute. Continuing; neuroboros may still find files via env var.")
        else:
            print("Could not guess a forrest dataset root under the provided folder. Continuing; neuroboros may still find files via env var.")

    # If nb has a module-level data root, try to set that too (best-effort)
    try:
        if hasattr(nb, "DATA_ROOT"):
            try:
                nb.DATA_ROOT = base
                print("Set nb.DATA_ROOT to", base)
            except Exception:
                pass
    except Exception:
        pass

    # Prepare output folder
    outdir = os.path.join(base, "results") if args.outdir is None else os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print("Results will be written to:", outdir)

    # Run the benchmark code (mirrors the code you provided)
    try:
        sids = dset.subjects
        print("Subjects:", sids)

        # ---- Load data once ----
        t_load = time.perf_counter()
        X_train, X_test = {}, {}
        Y_train, Y_test = {}, {}
        sls_data, dists_data = {}, {}
        radius = 20
        for lr in "lr":
            X_train[lr] = np.concatenate([dset.get_data(sids[0], "forrest", run_, lr) for run_ in [1, 2, 3, 4]], axis=0)
            Y_train[lr] = np.concatenate([dset.get_data(sids[1], "forrest", run_, lr) for run_ in [1, 2, 3, 4]], axis=0)
            X_test[lr] = np.concatenate([dset.get_data(sids[0], "forrest", run_, lr) for run_ in [5, 6, 7, 8]], axis=0)
            Y_test[lr] = np.concatenate([dset.get_data(sids[1], "forrest", run_, lr) for run_ in [5, 6, 7, 8]], axis=0)
            sls_data[lr], dists_data[lr] = nb.sls(lr, radius, return_dists=True)
        print(f"Data loaded in {time.perf_counter() - t_load:.2f}s")

        # ---- Alignment loop (repeated) ----
        all_times_l = []
        all_times_r = []
        all_times_total = []

        for rep in range(args.repeat):
            Ws = {}
            t_total = time.perf_counter()
            for lr in "lr":
                sls = sls_data[lr]
                dists = dists_data[lr]
                t_align = time.perf_counter()
                if args.backend == "python":
                    W = ha.searchlight_procrustes(
                        X_train[lr], Y_train[lr], sls, dists, radius
                    )
                else:
                    W = hac.searchlight_procrustes(
                        X_train[lr], Y_train[lr], sls, dists, radius,
                        backend=args.backend, n_jobs=args.n_jobs
                    )
                elapsed_lr = time.perf_counter() - t_align
                if lr == "l":
                    all_times_l.append(elapsed_lr)
                else:
                    all_times_r.append(elapsed_lr)
                Ws[lr] = W
            elapsed_total = time.perf_counter() - t_total
            all_times_total.append(elapsed_total)

            n_sls_l = len(sls_data["l"])
            n_sls_r = len(sls_data["r"])
            if args.repeat > 1:
                print(f"  Run {rep+1}/{args.repeat}: "
                      f"l={all_times_l[-1]:.2f}s ({n_sls_l} SLs), "
                      f"r={all_times_r[-1]:.2f}s ({n_sls_r} SLs), "
                      f"total={elapsed_total:.2f}s")
            else:
                print(f"  l hemisphere: {all_times_l[-1]:.2f}s ({n_sls_l} searchlights)")
                print(f"  r hemisphere: {all_times_r[-1]:.2f}s ({n_sls_r} searchlights)")

        if args.repeat > 1:
            med_l = np.median(all_times_l)
            med_r = np.median(all_times_r)
            med_total = np.median(all_times_total)
            print(f"  Median: l={med_l:.2f}s, r={med_r:.2f}s, total={med_total:.2f}s")

        # compute r_train0 and r_test0
        r_train0 = np.concatenate([
            np.mean(zscore(X_train[lr], axis=0) * zscore(Y_train[lr], axis=0), axis=0)
            for lr in "lr"
        ])
        r_test0 = np.concatenate([
            np.mean(zscore(X_test[lr], axis=0) * zscore(Y_test[lr], axis=0), axis=0)
            for lr in "lr"
        ])

        np.set_printoptions(formatter={"float": lambda x: f"{x:7.4f}"})
        print("Percentiles (train):")
        try:
            nb.percentile(r_train0)
        except Exception:
            print("nb.percentile(r_train0) failed; continuing.")
            traceback.print_exc()

        print("Percentiles (test):")
        try:
            nb.percentile(r_test0)
        except Exception:
            print("nb.percentile(r_test0) failed; continuing.")
            traceback.print_exc()

        # compute Yhat (optional)
        Yhat_train, Yhat_test = {}, {}
        for lr in "lr":
            Yhat_train[lr] = X_train[lr] @ Ws[lr]
            Yhat_test[lr] = X_test[lr] @ Ws[lr]

        # Save r_test0 and metadata
        out_npy = os.path.join(outdir, "r_test0.npy")
        out_npz = os.path.join(outdir, "forrest_results.npz")
        np.save(out_npy, r_test0)
        np.savez_compressed(out_npz,
                            r_test0=r_test0,
                            subjects=sids,
                            created=str(dt.datetime.now()))
        print("Saved r_test0 to:", out_npy)
        print("Saved compressed bundle to:", out_npz)

    except Exception:
        print("Benchmark run failed. Traceback:")
        traceback.print_exc()
        sys.exit(4)

    print("Done.")

if __name__ == "__main__":
    main()
