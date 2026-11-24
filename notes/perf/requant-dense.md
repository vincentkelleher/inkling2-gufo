# Dense-tensor requantization: measured, and it loses (2026-11-24)

The one route past the Q8_0 bandwidth bound: a decode step of `UD-Q4_K_XL` reads
~3.8 GB of weights, of which ~2.1 GB is the Q8_0 dense set (attention
q/k/v/gate/rel 1.01 GB, o-proj 0.36 GB, lm_head 0.49 GB). A bandwidth-only
estimate priced requantizing that set at 76–81 tok/s. This note replaces the
estimate with a measurement.

Tool: `tools/requant_dense.py` (new) — dequantizes the dense Q8_0 tensors row
chunk by row chunk, requantizes them, and copies every other tensor through as
raw bytes, so the output differs from the served file only in those tensors. The
source file is untouched.

```sh
uv venv /tmp/qvenv && uv pip install --python /tmp/qvenv/bin/python gguf numpy
/tmp/qvenv/bin/python tools/requant_dense.py \
  --src ~/Models/unsloth/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --dst ~/Models/unsloth/Inkling-2-Mini-GGUF-dense5/Inkling-2-Mini-UD-Q4_K_XL-dense5.gguf \
  --type q5_0
```

**Type choice is forced.** gguf-py ships an encoder only for the legacy 32-block
types: `Q4_K`, `Q5_K`, `Q6_K` and `IQ4_NL` raise `NotImplementedError` on
`quantize()`. `Q5_0` (5.5 bits) is the only sub-8-bit type in the extension's
accepted list (`F32 F16 BF16 Q8_0 Q4_K Q5_K Q6_K Q5_0 IQ4_NL IQ4_XS NVFP4`) with
an encoder, so that is what was built: 145 tensors, Q8_0 1.86 GB → Q5_0 1.20 GB,
file 17.63 GB → 17.21 GB.

## Measured (standalone, 32K ctx, 1 session, lookup on, pinned nonce)

| case | served Q8_0 file | dense Q5_0 file | delta |
| --- | --- | --- | --- |
| decode-short (47 tok) | 46.9 | **49.5** | +5.5 % |
| decode-4k | 44.4 | **46.5** | +4.7 % |
| decode-16k | 42.5 | **43.0** | +1.2 % |
| prefill-4k | 1,563 | **705** | −55 % |
| prefill-16k | 1,393 | **~643** | −54 % |

Through the llama-swap entry (temporarily pointed at the file, then reverted):
49.1 tok/s on a math prompt against 48.3 for the served file — neutral, inside
the ±10 % that the drafter's acceptance moves this prompt by.

## Why it does not work

- **The fast decode path is Q8_0 by construction.** `Engine` sets
  `f.proj`/`f.oproj`/`head_fast` only when `IsType(..., kTQ8_0)`, and
  `k_gemv_multi` hard-codes the 34-byte Q8_0 block plus int8 dot products. A
  Q5_0 dense file runs the generic per-tensor path for exactly the tensors that
  matter.
- **The generic decode path is compute-bound, not bandwidth-bound.** Cutting the
  dense read by ~0.66 GB (−18 % of the 3.76 GB step) buys +5 %, not +22 %: the
  dequant work eats the saved bytes. (For scale, forcing the whole decode onto
  the generic path with `INKLING2_SLOWDECODE=1` on the stock file reads
  14.4 tok/s.)
- **Prefill has no Q5_0 GEMM.** The prefill path's WMMA tiles are Q8_0/f16;
  Q5_0 falls back to dequant + gemm and loses 2.2x. That alone disqualifies the
  file for a serving entry, where a 4K prompt goes from 2.6 s to 5.8 s.

## Corrected prediction

The 76–81 tok/s figure assumed the bytes saved translate at Q8_0's achieved
bandwidth (203–224 GB/s of the 242 GB/s ceiling). Measured, requantization alone
gives **+5 % decode and −55 % prefill** and is rejected. Even with perfect
kernels the byte arithmetic caps it: 3.76 GB → ~2.6 GB is +45 %, not +70 %.

What would make the route real: Q4_K/Q5_K **fast** kernels — a multi-row GEMV
for the dense roles behind the same `DispatchType` gate as Q8_0, plus a WMMA
prefill path for the same types — and only then a requantized file. That is a
kernel project, not a file-conversion project, and it is the only route past
~60 tok/s on this model.

Not measured: KL. The change fails the speed and prefill gates first, so the
quality guardrail never comes into play. The 17 GB candidate was deleted after
the measurement; the script above rebuilds it in ~3 minutes.
