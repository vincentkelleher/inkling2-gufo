# Per-stage time map on 0.1.4 (2026-11-24)

Refreshes `notes/perf/map.md` (0.1.2, 2026-11-22) for the shipped code. Same
method: in-process `inkling2_bench` from a `BUILD_TESTING=ON` dev build, traced
with `rocprofv3 --kernel-trace`, roles from `tools/prof_stage.py` (which now has
a dspark role).

Build and trace recipe (`~/inkling2-work` keeps the incremental tree):

```sh
rsync -a --delete --exclude .git --exclude lab --exclude flake.lock \
  ~/Source/inkling2-gufo/ ~/inkling2-work/ext-build/
cd ~/inkling2-work/ext-build && MODEL_DIR=~/Models \
  tools/dev-env.sh "cd /work/ext-hip && ninja -j20 gufo inkling2_bench"

R=/nix/store/r3cfrx1cprrvnnnxgkd0lf9lyl9zjfnn-rocprofiler-sdk-10.1/bin/rocprofv3   # 10.1!
$R --kernel-trace -f csv -o d014 -- inkling2_bench --model …/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --text /work/perf/corpus.txt --ctxs 1024 --decode 48 --ctx 32768
python3 tools/prof_stage.py d014_kernel_trace.csv --decode 8
```

Trap worth recording: the store's `rocprofiler-sdk-7.2.4` runs, reports success
and profiles nothing — the app prints `no ROCm-capable device is detected` and
the trace is empty. `rocprofiler-sdk-10.1` (the ROCm the closure is built
against) works. Two wasted runs.

## Decode step, 1K context, batch 1 (last 8 steps of 48)

Wall 23,452 µs/step = 42.6 tok/s. Kernels 21,040 µs (89.7 %), **not in kernels
2,412 µs (10.3 %)** over **671 launches/step** (~3.6 µs of gap each).

| role | launches | µs | % | MB read | GB/s |
| --- | --- | --- | --- | --- | --- |
| MoE gate+up + SITU (NVFP4) | 44 | 4,983 | 23.7 | 1,051 | 211 |
| dense attn projections q/k/v/gate/rel (Q8_0) | 48 | 4,654 | 22.1 | 1,008 | 216 |
| MoE down (NVFP4) | 44 | 2,577 | 12.2 | 525 | 204 |
| lm_head (Q8_0) + softcap | 1 | 2,189 | 10.4 | 490 | 224 |
| dense o-proj (Q8_0) | 48 | 1,753 | 8.3 | 361 | 206 |
| attention split-K + reduce + KV prep | 144 | 1,444 | 6.9 | — | — |
| MoE gate+up (Q8_0) | 4 | 817 | 3.9 | 180 | 221 |
| attention-residual mix (+norm, +post-conv) | 120 | 811 | 3.9 | — | — |
| loop controllers (F32 gate + apply) | 96 | 653 | 3.1 | 227 | 348¹ |
| MoE down (Q8_0) | 4 | 425 | 2.0 | 90 | 212 |
| router (F32 GEMV + biased top-8) | 48 | 399 | 1.9 | 57 | 142 |
| norms, embed, sconv, post-conv | 63 | 325 | 1.5 | — | — |
| copies/fills | 7 | 10 | 0.0 | — | — |

¹ above the 242 GB/s stream ceiling because the controllers' rows stay in cache
across the 96 launches.

**The weights are the step.** Summing the weight-bound roles: 17,398 µs reading
3,705 MB = **213 GB/s, 88 % of the 242 GB/s ceiling**. Non-weight kernels are
3,642 µs (attention 1,444 + mix 811 + controllers 653 + router 399 + norms 325).
Headroom to the 60.6 tok/s structural bound (16.5 ms) is therefore *not* in these
kernels: 23,452 − 16,500 = 6,952 µs, of which 2,412 is gaps, 3,642 non-weight
kernels, and the remaining ~900 is the gap between 213 and 242 GB/s.

Ranked decode candidates, with their measured size:

1. **Launch cadence / gaps — 2,412 µs (10.3 %).** 671 launches of which 144 are
   attention and 120 are the residual mix. `SplitFor` (engine.cpp) gives
   `chunk = clamp(bucket_ctx/32, 128, 2048)` → at a 2,048 bucket, 16 splits per
   global visit; the mix kernels run per visit as well. Fusing the mix into the
   projection launch, or cutting the split count where the GPU is already full,
   is the only lever that can move ~10 % without touching bytes.
2. **Attention at depth — 1,444 µs at 1K, and it grows.** It reads f16 KV: at 32K
   the 8 global visits read 1.07 GB/token. q8 KV (not implemented) would halve
   that and is the single largest structural lever left at long context.
3. **Router at 142 GB/s — 399 µs.** The worst bandwidth of any weight kernel;
   it reads 57 MB of F32 per step. Narrowing the router's weight type is a
   precision-cliff change (measured before: f16 there costs KL 1.6e-3), so this
   is listed for completeness, not as a recommendation.

## Prefill, 4,096-token window

Shares of the traced window (1,991 kernels, 1,318,404 µs in kernels, 0.9 % gaps):

| role | kernels | µs | % | µs/token |
| --- | --- | --- | --- | --- |
| prefill GEMMs (WMMA) | 384 | 848,643 | 64.4 | 207 |
| prefill routing / histogram / conversions | 600 | 126,468 | 9.6 | 31 |
| loop controllers (F32 gate + apply) | 48 | 70,024 | 5.3 | 17 |
| mix_k | 72 | 64,822 | 4.9 | 16 |
| attn_prefill | 48 | 62,503 | 4.7 | 15 |
| norms, embed, sconv, post-conv | 571 | 57,924 | 4.4 | 14 |
| k_gate_wh | 48 | 53,957 | 4.1 | 13 |
| attention-residual mix | 47 | 22,448 | 1.7 | 5 |
| copies/fills | 124 | 9,969 | 0.8 | 2 |
| attention split-K + reduce + KV prep | 48 | 1,641 | 0.1 | 0 |

Caveat stated rather than papered over: the traced window is 1.33 s while the
bench reports 2.73 s for the same 4,096-token prefill, so the table explains the
shares of the traced half, not the whole measured time. The shares match the
0.1.2 map (GEMM 61 % then, 64 % now) — the 0.1.4 kernel work moved the ctl/gate
entries *into* GEMM time, as expected.

Ranked prefill candidates:

1. **GEMM 64.4 % at ~24 of 55 TFLOPS f16** — the wave-tile rewrite (the lab's
   H2–H4 conclusion). Big, structural, and the only path to 2,000 tok/s.
2. **Routing/histogram/conversions 9.6 %** — 600 kernels, 31 µs/token of
   non-GEMM work with no FLOPs in it. The cheapest real prefill win available:
   find out how much of that is `k_cvt_f16` on buffers that are converted once
   per loop pass, and fuse or hoist it.
3. **Small per-visit kernels — controllers 5.3 % + mix_k 4.9 % + k_gate_wh 4.1 %
   = 14.3 %** at 13–17 µs/token each. These are the latency-bound launches the
   0.1.2 map already named; the 0.1.4 tiles helped the controller entries and
   there is still 14 % sitting in three kernels.

## What this means for the goal

Decode is at 85 % of a bound that is itself 74 % weight reads at 88 % of DRAM
bandwidth; no kernel change available in a single iteration can move it more than
~10 %, and only by attacking gaps/fusion. Prefill has a 36 %-wide GEMM target and
a 9.6 % conversion bucket that looks like the cheapest 5–9 % in the whole map.
Both tasks stay open until one of those is landed with numbers.

## Follow-up: are the 2.4 ms of gaps launch overhead? No.

Interleaved test, `decode-4k`, one fresh load per side, `INKLING2_LOOKUP=0`,
two pairs: graphs on 44.0 / 44.0 tok/s, `INKLING2_NOGRAPH=1` 44.1 / 44.2. The
captured-graph path is a no-op for speed here, so the 2,412 µs between kernels
are *dependency stalls* (a 5–25 µs kernel the next launch waits on), not launch
cost. That removes "turn on HIP graphs" from the candidate list and leaves fusion
of the small dependent kernels as the only way to touch those 10 %.

Per-kernel counts in one decode step (671 launches, from the same trace): 44
`k_moe_gu<1,…>` (113 µs each), 44 `k_moe_down_e<1,…>` (58 µs), 48
`k_gemv_multi` (98 µs) for the attn projections, 48 `k_gemv<0,3072,…>` (37 µs,
o-proj), 1 `k_gemv<0,2304,…>` (2,198 µs, lm_head), 48 `attn_decode<4>` (24 µs),
**119 `k_mix`** (6.7 µs), 48 `k_router` (8.3 µs), 48 `k_ctl_gate` + 48
`k_ctl_apply` (6.9 + 6.6 µs), 49 `k_rms` (5.8 µs). The 6–9 µs kernels are 320
launches of the step and cost ~1.7 ms of work plus their share of the stalls.

Prefill, same file, by name: `k_gemm4` variants 56.4 % (88 + 88 + 480 + 96 + 8
launches), `attn_prefill` 6.3 %, `k_ctl_apply_w` 4.8 %, `mix_k` 4.4 %, `k_gate_p`
3.7 % (224 launches), `k_gate_wh` 3.7 %, `k_sum_pairs` 2.5 %.

## Next iteration, scoped from the map (not from guessing)

**Decode — fuse the small dependent kernels (target: the 320 launches of 6–9 µs).**
One step is 671 launches; 320 of them are `k_mix` (119), `k_ctl_gate` (48),
`k_ctl_apply` (48), `k_router` (48), `k_rms` (49), each doing 5–9 µs of work.
With graphs proven to be a no-op and the stalls measured at 2.4 ms, the win is
fewer, fatter kernels. Smallest credible first step: fold the two `k_mix` calls
of a visit into one launch (a 2-D grid over both outputs, same arithmetic per
row) and measure with `ab.sh --cases decode-4k,decode-short --reps 3
--mode lookup-off/dspark-off`. Expected order: 1–3 % (≈200–400 µs of the 23.4 ms
step). Do **not** start with the controllers: they are sequential (gate → apply)
and cannot be fused without a persistent-kernel rewrite.

**Prefill — the conversion/routing bucket, then the GEMM.**
`k_gate_p` (224 launches), `k_sum_pairs` (96) and the conversion kernels are
~10 % of the window at 13–31 µs/token with no FLOPs in them. First question to
answer from the code: how many of those runs are per loop pass over a buffer
that is identical across passes — if any, hoisting them is a cheap 2–5 %.
Only after that, the GEMM: `k_gemm4` variants are 56 % of the window at ~24 of
55 TFLOPS and need the wave-tile rewrite (lab H2–H4), which is a project.

**Gate for both:** `tests/perf/ab.sh` against the shipped image, `--reps 3`,
pinned nonce; accept only if the pair ratios agree in sign and clear 1 % (decode)
/ 1.5 % (prefill). Quality: mean KL ≤ 1e-3, ≥ 98 % top-1 via the oracle tests if
the numerics change at all.
