# Round report — 2026-11-24: 0.1.4 to its ceilings, what moved and what did not

Image `localhost/gufo-runtime:inkling2` = Gufo v0.35.0 (`0260e11`) + extension
`24f8342` (0.1.4). `UD-Q4_K_XL` Mainline_Support. Strix Halo (gfx1151, 128 GiB).
**No source change was accepted this round**, so the image is the same one built
at the start; what changed is the llama-swap entry (drafter on) and the
measurement scaffolding. Numbers below are from `tests/perf/bench.py` and
`tests/perf/workloads.py`; the protocol and its noise floor are in
`notes/perf/baseline-014.md`.

## Achieved against the ceilings

| metric | before this round | after | measured ceiling | share of ceiling |
| --- | --- | --- | --- | --- |
| decode, entry, 4K ctx | 45.6 | **51.8** | 60.6 (kernel-only, weights at 242 GB/s) | 85 % |
| decode, entry, short | 48.9 | **52.6** | 60.6 | 87 % |
| decode, plain path, 4K | 44.1 | 44.1 | 60.6 | 73 % |
| decode, width-4 verify | — | — | 93 (43.2 ms verifier) | — |
| prefill, 4K | 1,537 | **1,530–1,554** | ~2,000 (GEMM at 55 TFLOPS f16) | ~77 % |
| prefill, 16K | 1,393–1,548 | 1,393–1,548 | ~2,000 | ~70–77 % |
| batch-2 aggregate | 68.3 | 66–75 | — | — |

Where the decode gain came from, in one line: the DSpark drafter, wired into the
entry (`--speculative dspark`), not from any kernel change. It is worth +4 % on
chat, +22 % on math, +7.5 % on code and +97 % on copy-verbatim, and it still pays
at 16K of context (`notes/perf/workloads-014.md`). The plain path is unchanged
and the prefill path is unchanged.

## Residual gap and its reason

- **Decode, 51.8 → 60.6 (85 %).** A step still reads ~3.8 GB of weights at an
  end-to-end 170 GB/s while the big kernels individually reach 203–224 GB/s of
  the 242 GB/s ceiling; the difference is attention/KV work, the small dependent
  kernels (norm, router, controllers) and ~2.4 ms of inter-kernel gaps measured
  in the 0.1.2 map. Closing it needs either q8 KV (not implemented; at 32K the
  8 global visits read 1.07 GB of f16 KV per token) or fewer bytes per step.
- **Decode past 60.6.** Only fewer bytes per step can do it. The dense Q8_0 set
  is 2.13 GB of the 3.76 GB read, and requantizing it is measured **+5 % decode
  and −55 % prefill** (`notes/perf/requant-dense.md`) because the fast kernels
  are Q8_0 by construction. The route past the bound is Q4_K/Q5_K *fast* GEMV and
  WMMA prefill kernels, then a requantized file: byte arithmetic caps that at
  ~+45 % (3.76 → 2.6 GB), i.e. ~75 tok/s.
- **Prefill, ~1,540 → 2,000 (77 %).** 61 % of a prefill token is GEMM at ~24 of
  55 TFLOPS and the small kernels are latency-bound at 1–5 % of hardware limits.
  Every cheap knob is measured dead (this round: ubatch 4096 unproven, ubatch
  8192 does not load, prefill-chunk 8192 inside noise; previous round: MoE tile
  128, ubatch 1024/4096, f16 WMMA router/sink for KL). The remaining fix is the
  kernel lab's H2–H4 conclusion: a wave-tile rewrite of the prefill GEMM.

## Branches and PRs

| branch | PR | contents |
| --- | --- | --- |
| `perf/014-baseline` | [#7](https://github.com/headbouyJB/inkling2-gufo/pull/7) (draft) | `bench.py`, `matrix.sh`, `ab.sh`, `prof_stage.py`, `baseline-014.md`, four-mode baseline JSONs |
| `perf/speculation-defaults` | [#8](https://github.com/headbouyJB/inkling2-gufo/pull/8) (draft) | `workloads.py`/`.sh`, `workloads-014.md`, drafter table and policy sweep |
| `perf/requant-dense` | none (rejected) | `tools/requant_dense.py`, `requant-dense.md` |
| `perf/prefill-knobs-014` | none (rejected) | `prefill-knobs-014.md` |

## Every dead end, one line each

| attempt | number that killed it |
| --- | --- |
| dense Q8_0 → Q5_0 requant | decode +5 %, prefill −55 % (705 vs 1,563), entry neutral |
| `INKLING2_SLOWDECODE=1` (whole step generic) | 14.4 tok/s vs 43.9 — bounds any non-fast type |
| `INKLING2_UBATCH=4096` | +2.2 % at 4K, −0.9 % at 16K; three-pair recheck 1.033 / 1.036 / 0.981 |
| `INKLING2_UBATCH=8192` | does not load |
| `--prefill-chunk 8192` | +1.3 % at 4K, 0 % at 16K |
| `INKLING2_DSPARK_FIXED=3` | +6 % math, −6 % code, −11 % copy |
| `INKLING2_DSPARK_FIXED=5` | −17 % chat |
| `INKLING2_DSPARK_TOPK=0` (exact head) | −13 to −35 % on every workload |
| `INKLING2_DSPARK_MINCTX=1` | identical to the default |
| per-session drafter memory estimate (README) | measured +629 MiB, not ~2 GB |

## Still open

Task 5 (decode loop) has its analysis but no iteration: the decode budget is
written down in `baseline-014.md` and above, and the two candidate directions are
q8 KV (kernel project: attention kernels + state layout + the lookup/dspark
rollback snapshots) and the launch-gap/small-kernel fusion (~2.4 ms of ~22 ms).
Task 4 (prefill loop) has its knob sweep recorded and no accepted change; it
needs the GEMM wave-tile rewrite to move.
