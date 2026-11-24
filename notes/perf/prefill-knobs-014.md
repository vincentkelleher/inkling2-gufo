# Prefill knobs re-measured on 0.1.4 — all inside the noise, nothing accepted

2026-11-24, image `localhost/gufo-runtime:inkling2` (Gufo `0260e11` + `24f8342`),
`UD-Q4_K_XL`, `--sessions 1 --context 32768 --cache-ram-bytes off`, one fresh
load per row, `tests/perf/bench.py` with pinned nonce (`pf-sweep`), 2 reps.

| config | prefill-4k | prefill-16k |
| --- | --- | --- |
| default (`INKLING2_UBATCH` 2048, `--prefill-chunk 4096`) | 1,558.8 | 1,545.1 |
| `INKLING2_UBATCH=4096` | 1,593.3 (+2.2 %) | 1,530.8 (−0.9 %) |
| `INKLING2_UBATCH=8192` | **does not load** (server never becomes ready) | — |
| `--prefill-chunk 8192` | 1,579.0 (+1.3 %) | 1,545.0 (0.0 %) |

The `UBATCH=4096` row looked like +3 % at 4K, so it was re-measured as a
pair-wise comparison, three times, one load each side, `prefill-4k` only:

| rep | default | ubatch 4096 | ratio |
| --- | --- | --- | --- |
| 1 | 1,537.0 | 1,588.1 | 1.033 |
| 2 | 1,540.0 | 1,595.0 | 1.036 |
| 3 | 1,581.2 | 1,551.5 | 0.981 |

Two of three pairs favour 4096 and the third reverses; the order was not flipped
between sides, so the second measurement of each pair is systematically the
warmer one. **Verdict: unproven, rejected.** It is the same knob the previous
round rejected as "inside noise" (`perf/prefill-ubatch`, never merged), and this
run does not overturn that.

Dead ends recorded:

- `INKLING2_UBATCH=8192` — the server never comes up; leaving 8K-token micro-batches
  to the host's fit check. One line, no number to report beyond "refuses to load".
- `--prefill-chunk 8192` — +1.3 % at 4K, 0 % at 16K: inside the 1.5 % prefill noise
  floor measured in `baseline-014.md`.

## Where the remaining prefill time is (and why no fix landed this round)

No fresh trace was obtained: the runtime image ships no `rocprofv3` and building
the `BUILD_TESTING=ON` profiler path (`inkling2_bench` + `tools/prof_stage.py`)
did not fit the remaining budget. The last measured map is the 0.1.2 one
(`git-archive` in `~/Inference/inkling-2-mini`, `notes/perf/map.md`, 692.5 us per
prefill token) and 0.1.4 has since changed exactly three of its entries — the
controller down-projection tiles (`k_ctl_apply_w` 78 → 52 → 34 us/token),
`k_gate_p` float4 LDS tiles (73.8 → 63.6) and `k_ctl_apply_p` 32-token blocks
(93 → 78) — all landed upstream as PR #4. What is left is what that map already
named: 61 % GEMM at ~24 of 55 TFLOPS f16, and small kernels that are
latency-bound at 1–5 % of hardware limits.

The only remaining cheap knobs are the ones measured above, and the real fix is
the one the kernel lab's H2–H4 notes already concluded: a wave-tile rewrite of
the prefill GEMM (bigger wave tiles, tensor-core shapes) with new numerics. That
is a project, not an iteration, and it is where the next round should start.
