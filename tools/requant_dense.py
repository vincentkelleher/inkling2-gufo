#!/usr/bin/env python3
"""Requantize the DENSE tensors of an Inkling 2 Mini GGUF, leaving the rest byte-identical.

Why: a decode step of UD-Q4_K_XL reads ~3.8 GB of weights, of which ~2.1 GB is
Q8_0 dense (attention q/k/v/gate/rel, o-proj, lm_head). The extension's fast
decode kernels are Q8_0-only by construction (`k_gemv_multi` hard-codes the
34-byte Q8_0 block and int8 dot products, and `Engine::BatchCapable` requires
Q8_0 for exactly these tensors), so this script exists to measure what 4-bit
dense weights would be worth once such kernels exist -- not to produce a faster
file today. See notes/perf/requant-dense.md for the measured answer.

What it does, per tensor:
  * dense Q8_0 tensor whose row length is a multiple of 256 -> dequantize to f32
    in row chunks, quantize to Q4_K, write;
  * everything else -> copied through as raw bytes, so the output is the same
    file apart from those tensors.

Usage: requant_dense.py --src FILE --dst FILE [--type q4_K|q5_K|q6_K|q4_0]
                        [--only REGEX] [--dry-run]
Requires the `gguf` python package (gguf-py) and numpy.
"""
import argparse
import os
import re
import sys

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize, GGML_QUANT_SIZES

DEFAULT_TYPE = "q4_K"
# The tensors the fast decode path requires: attn q/k/v/gate/rel, o-proj, lm_head.
DEFAULT_ONLY = r"\.attn_(q|k|v|gate|rel|output)\.weight$|^output\.weight$"

QMAP = {"q4_K": Q.Q4_K, "q5_K": Q.Q5_K, "q6_K": Q.Q6_K, "q4_0": Q.Q4_0, "q5_0": Q.Q5_0}
# gguf-py ships an ENCODER only for the legacy 32-block types (Q4_0, Q5_0, Q8_0);
# the K-quants and IQ types are decode-only, so q4_K/q5_K/q6_K above are accepted
# by the parser and fail at quantize() with NotImplementedError. Q5_0 is the only
# 5-bit type the extension accepts (F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K/Q5_0/IQ4_NL/
# IQ4_XS/NVFP4), which is why the measurement below uses it.


def row_bytes(k, qtype):
    blk, ts = GGML_QUANT_SIZES[qtype]
    assert k % blk == 0, f"row length {k} is not a multiple of {blk}"
    return k // blk * ts


def requant(t, qtype, rows_per_chunk):
    """Q8_0 tensor -> raw bytes of `qtype`, row chunk by row chunk."""
    k = int(t.shape[0])
    n = int(t.shape[1]) if len(t.shape) > 1 else 1
    src_rb = row_bytes(k, Q.Q8_0)
    raw = np.asarray(t.data).view(np.uint8).reshape(n, src_rb)
    out = []
    for start in range(0, n, rows_per_chunk):
        chunk = raw[start:start + rows_per_chunk]
        f32 = dequantize(chunk.reshape(-1), Q.Q8_0).astype(np.float32).reshape(chunk.shape[0], k)
        out.append(quantize(f32, qtype).view(np.uint8))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--type", default=DEFAULT_TYPE, choices=sorted(QMAP))
    ap.add_argument("--only", default=DEFAULT_ONLY, help="regex over tensor names")
    ap.add_argument("--rows-per-chunk", type=int, default=256, help="bounds the f32 working set")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qtype = QMAP[args.type]
    only = re.compile(args.only)
    reader = GGUFReader(args.src, mode="r")
    arch = reader.get_field("general.architecture").contents()
    print(f"src {args.src}\n  arch {arch}, {len(reader.tensors)} tensors")

    todo, keep = [], []
    for t in reader.tensors:
        name = t.name
        if t.tensor_type == Q.Q8_0 and only.search(name):
            k = int(t.shape[0])
            blk, _ = GGML_QUANT_SIZES[qtype]
            if k % blk:
                keep.append((name, t.tensor_type.name, f"row {k} not a multiple of {blk}"))
                continue
            todo.append(t)
        else:
            keep.append((name, t.tensor_type.name, "kept"))

    src_b = sum(int(t.data.nbytes) for t in todo)
    new_b = sum(int(requant(t, qtype, args.rows_per_chunk).nbytes) if args.dry_run else
                (int(t.shape[1] if len(t.shape) > 1 else 1) * row_bytes(int(t.shape[0]), qtype))
                for t in todo)
    print(f"  requantize {len(todo)} tensors {src_b/1e9:.2f} GB -> {new_b/1e9:.2f} GB ({args.type})")
    print(f"  keep        {len(keep)} tensors as-is")
    if args.dry_run:
        for name, qt, why in keep:
            if why != "kept":
                print(f"    left alone: {name} ({qt}) — {why}")
        return

    writer = GGUFWriter(args.dst, arch)
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        vtype = field.types[0]
        try:
            writer.add_key_value(key, field.contents(), vtype)
        except Exception as exc:  # noqa: BLE001 - report and continue, metadata matters less than the tensors
            print(f"  warning: metadata {key} not copied ({exc})", file=sys.stderr)

    # gguf-py's add_tensor takes the BYTE shape for a raw quantized tensor and
    # converts it to the logical shape itself (`quant_shape_from_byte_shape`), so
    # pass (rows, row_bytes), not (row_len, rows).
    for t in reader.tensors:
        k = int(t.shape[0])
        n = int(t.shape[1]) if len(t.shape) > 1 else 1
        if t in todo:
            data = requant(t, qtype, args.rows_per_chunk)
            writer.add_tensor(t.name, data, raw_dtype=qtype)  # data is already in byte shape
            print(f"  {t.name}: {t.tensor_type.name} -> {args.type} ({data.nbytes/1e6:.1f} MB)", flush=True)
        elif t.tensor_type in (Q.F32, Q.F16):
            dtype = np.float32 if t.tensor_type == Q.F32 else np.float16
            writer.add_tensor(t.name, np.asarray(t.data, dtype=dtype))
        else:
            # Byte shape straight from the reader: expert tensors are 3-D, so
            # recomputing (rows, row_bytes) from the logical shape gets it wrong.
            writer.add_tensor(t.name, np.asarray(t.data).view(np.uint8), raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"wrote {args.dst} ({os.path.getsize(args.dst)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
