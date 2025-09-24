#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import numpy as np

def to_index_lists(mat, threshold=None):
    """
    Convert a sparse matrix to a list of index lists per row.
    If threshold is given, select only columns where data > threshold.
    """
    try:
        import scipy.sparse as sp
    except Exception as e:
        raise RuntimeError("scipy is required. Please run `pip install scipy` and try again.") from e

    csr = mat.tocsr()  # Convert any format to CSR
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data

    out = []
    if threshold is None:
        for i in range(csr.shape[0]):
            sl = slice(indptr[i], indptr[i+1])
            out.append(indices[sl].tolist())
    else:
        thr = float(threshold)
        for i in range(csr.shape[0]):
            sl = slice(indptr[i], indptr[i+1])
            if sl.start == sl.stop:
                out.append([])
                continue
            idx_slice = indices[sl]
            dat_slice = data[sl]
            mask = dat_slice > thr
            out.append(idx_slice[mask].tolist())
    return out

def try_load(path):
    """
    Try to load a sparse matrix from various formats automatically.
    1) scipy.sparse.load_npz
    2) numpy.load(..., allow_pickle=True) (sparse object/dict)
    3) scipy.io.mmread (Matrix Market)
    4) Call load_spmat/read_spmat from repo utils (dataset_io, etc.)
    5) Simple CSR binary heuristic
    """
    # 1) Standard npz
    try:
        import scipy.sparse as sp
        return sp.load_npz(path)
    except Exception:
        pass

    # 2) numpy + pickle
    try:
        ld = np.load(path, allow_pickle=True)
        if hasattr(ld, "files") and ld.files:
            obj = ld[ld.files[0]]
            try:
                import scipy.sparse as sp
                if sp.issparse(obj):
                    return obj
            except Exception:
                pass
            if isinstance(obj, dict):
                import scipy.sparse as sp
                data = obj.get("data")
                indices = obj.get("indices")
                indptr = obj.get("indptr")
                shape = tuple(obj.get("shape", ()))
                if data is not None and indices is not None and indptr is not None and shape:
                    return sp.csr_matrix((data, indices, indptr), shape=shape)
    except Exception:
        pass

    # 3) Matrix Market
    try:
        from scipy.io import mmread
        return mmread(path).tocsr()
    except Exception:
        pass

    # 4) Project utils (dataset_io.py, etc.)
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            "dataset_io",
            "benchmark.dataset_io",
            "yfcc.benchmark.dataset_io",
        ]
        for mod in candidates:
            try:
                m = __import__(mod, fromlist=["*"])
                for fn in ["load_spmat", "read_spmat", "load_sparse_matrix", "read_sparse_matrix"]:
                    if hasattr(m, fn):
                        return getattr(m, fn)(path)
            except Exception:
                continue
    except Exception:
        pass

    # 5) Simple CSR binary heuristic (little-endian)
    try:
        import scipy.sparse as sp
        with open(path, "rb") as f:
            buf = f.read()
        if len(buf) >= 24:
            off = 0
            rows, cols, nnz = np.frombuffer(buf[off:off+24], dtype="<i8", count=3)
            off += 24
            if 0 < rows < 10**9 and 0 < cols < 10**9 and 0 <= nnz < 10**12:
                need_indptr = (rows + 1) * 8
                if off + need_indptr <= len(buf):
                    indptr = np.frombuffer(buf[off:off+need_indptr], dtype="<i8", count=rows+1)
                    off += need_indptr
                    need_indices = nnz * 4
                    if off + need_indices <= len(buf):
                        indices = np.frombuffer(buf[off:off+need_indices], dtype="<i4", count=nnz)
                        off += need_indices
                        rem = len(buf) - off
                        if rem >= nnz:
                            data = np.frombuffer(buf[off:off+nnz], dtype="|u1", count=nnz)
                        else:
                            data = np.ones(nnz, dtype=np.uint8)
                        mat = sp.csr_matrix((data, indices, indptr), shape=(rows, cols))
                        return mat
    except Exception:
        pass

    raise RuntimeError("Unknown .spmat format: all standard loaders/heuristics failed.")

def save_output(label_lists, path):
    """Determine save format by extension: .json / .jsonl / .txt|.tsv"""
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(label_lists, f, ensure_ascii=False)
        print(f"Saved JSON: {path}")
    elif path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for lst in label_lists:
                f.write(json.dumps(lst, ensure_ascii=False) + "\n")
        print(f"Saved JSONL: {path}")
    else:
        # Default: TSV (row_id \t 1,3,5)
        with open(path, "w", encoding="utf-8") as f:
            for i, lst in enumerate(label_lists):
                f.write(f"{i}\t{','.join(map(str, lst))}\n")
        print(f"Saved TXT/TSV: {path}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert one-hot sparse matrix (.spmat, etc.) to row-wise index lists"
    )
    ap.add_argument("input", help="Input .spmat/.npz/.mtx file path")
    ap.add_argument("--out", help="Output file path (.json | .jsonl | .txt/.tsv). If not specified, preview to stdout")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Only accept labels where data > threshold (default: not used)")
    ap.add_argument("--preview", type=int, default=10,
                    help="Number of preview rows to stdout (default: 10)")
    ap.add_argument("--quiet", action="store_true",
                    help="Minimize loading/statistics logs")
    return ap.parse_args()

def main():
    args = parse_args()

    mat = try_load(args.input)

    # Loading info
    if not args.quiet:
        try:
            import scipy.sparse as sp
            sparse_type = type(mat).__name__
            print(f"Loaded matrix type: {sparse_type}")
            print(f"Shape: {mat.shape}")
            if hasattr(mat, "nnz"):
                print(f"nnz: {mat.nnz}")
        except Exception:
            pass

    labels = to_index_lists(mat, threshold=args.threshold)

    if args.out:
        save_output(labels, args.out)
    else:
        # Preview to stdout
        n = min(args.preview, len(labels))
        for i in range(n):
            print(f"row {i}: {labels[i]}")
        if not args.quiet:
            print(f"(Preview {n}/{len(labels)} rows) -- To save to file, specify --out path.")

if __name__ == "__main__":
    main()