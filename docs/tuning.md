# Tuning record

What moved the numbers after M2's cutover, what did not, and why. Every step
kept the bench WAV's sha256 (`4cb7232a…`): all of this is arithmetic-order-
preserving. Numbers are `navi-tts bench` on the XTX, streaming configuration,
`bench/results.jsonl` has the rows.

| step | commit | frame ms | TTFA ms | RTF |
|---|---|---|---|---|
| M2 baseline | `1bb04b9` | 8.33 | 102 | 0.129 |
| vocoder batching (4 first, then 16; max for whole-body) | `7e9364b` | 8.31 | 66 | 0.125 |
| sampler: radix select + draw over candidates | `c3dac3d` | 7.56 | 63 | 0.116 |
| matvec 2 rows × 4 loads in flight per wave | `c8fe709` | 6.46 | 59 | 0.102 |
| barrier on one monotonic counter | `016acb3` | 6.24 | 58 | 0.099 |
| conv kernel: conflict-free weight tile, vector staging, static dilation | `90c88b8` | 6.13 | 41 | 0.088 |
| prefill: skinny GEMM, lane per prompt row | `52fcd7d` | 6.13 | 35 | 0.086 |
| prefill: q/k/v and gate/up as one GEMM each | `86f6372` | 6.13 | 34 | 0.086 |

Whole-body requests (the queue) sit ~0.004 below the streaming RTF.

The conv change moved everything but the frame: vocoder 1.30 -> 0.62 ms/frame,
prefill 25.7 -> 14.2 ms. One kernel serves the vocoder's convs and GEMMs, the
talker prefill and the speaker encoder, so a fix aimed at the k=7 convs
landed on all of them.

## How the frame was measured

- `rocprofv3 --kernel-trace --stats` on a `synth` run: the frame kernel is 81 %
  of GPU time, the vocoder's `k_conv` 17 %, everything else 1.4 %.
- `-DNAVI_FRAME_PROFILE` (a separate build dir): block 0 timestamps every grid
  barrier; `Frame` prints a per-phase table for frame 8. Per frame, before the
  barrier change: matvec phases 4.67 ms for ~3.3 GB of weights (707 GB/s),
  barrier waits 0.95 ms over 700 barriers, attention 0.35, sampling 0.22,
  norm/rope/embed 0.11.
- `bench/micro/matvec.hip`: the frame's matvec at the real grid from a 500 MB
  rotation (the 96 MB Infinity Cache flatters anything smaller). `bench/micro/
  sampling.hip`: the sort/draw pieces.

A frame streams ~3.3 GB (talker 0.89, code predictor 15 × 157 MB): at 6.24 ms
that is ~530 GB/s end to end, ~707 inside the matvec phases. The card sustains
~745–880 with a plain streaming kernel.

## What did not help (measured, reverted)

- **Balanced contiguous row partition** and a **fused q/k/v matvec**: block 0's
  phases shortened, its barrier waits lengthened by the same amount. A matvec
  phase ends when the memory system finishes, not when the last wave does;
  row assignment is not the lever.
- **norm/rope folded into attend** (one barrier per layer fewer): slower - the
  q-norm, k-norm, v and score latency chains serialised inside 16 blocks cost
  more than the barrier that let 32 blocks run them in parallel - and not
  bit-exact.
- **More rows per wave (R8)** and **one row with 8 loads (R1 U8)**: both slower
  than R2 U4 in the microbench.
- On RDNA3 `clock64()` does not count shader cycles; spin loops use
  `wall_clock64()` (the hog learned this the hard way).
- **Register-blocking the conv kernel (8×4, 8×8, 16×4 thread tiles)**: the
  premise was wrong. The kernel was not LDS-bandwidth-bound; its weight loads
  were 16-way bank-conflicted (56-dword lane stride), its fully unrolled 16×7
  inner loop cost 154 VGPRs for 8 accumulators, and its staging spent more
  instructions than its FMAs. Every wider tile spilled and ran slower. With
  those three fixed the 4×2 tile is within noise of the wider ones; a
  64×128 8×4 tile wins only on dilation 9 at 96-192 channels (~0.03 ms/frame,
  not worth a second instantiation). `bench/micro/conv7.hip`.
- **Unrolling the conv's channel-chunk loop** at all: u1 < u2 < u4 < u16, every
  step. The unrolled loads hoist into registers and occupancy pays.
- **Streaming the prefill GEMMs at the matvec's ~700 GB/s.** Not reachable
  under the bit-exact rule at T = 10: one lane must own each (t, co) dot
  product end to end, so a GEMM has Cout/R waves of dependent chains - 1-2
  per SIMD - and they stall on LDS reads (273 `s_waitcnt` per 2952 ISA
  instructions). Four mappings (lane per row with f16 or f32 tiles, lane per
  channel, quad-coalesced with DPP exchange), deeper prefetch, register
  double-buffering and occupancy hints all sit at 150-240 GB/s where pure
  streams do 450-730. Fusing q/k/v and gate/up measured ~15 % on top.
  `bench/micro/gemm_skinny.hip`.

## What is left

Ordered, with gates and expected numbers: `docs/tuning-todo.md`.

- **Vocoder**, now 0.62 ms/frame: the k=7 group is ~19 of the ~43 ms per 70
  frames and runs at 13-19 TFLOP/s on the wide layers, roughly half of the
  card's single-issue f32 rate. What is left there is VALU issue (the f16
  unpack per tap, ~25 % of the loop) and the two `__syncthreads` per 16-
  channel chunk. ~2 % of RTF at most; not next.
- **Prefill** (7.0 ms, 21 % of TTFA): the GEMMs are ~4.8 of it at 150-240
  GB/s, the ceiling for a bit-exact 10-row GEMM (above); q/k/v and gate/up
  are fused (the arena places named groups adjacently). Left: the non-GEMM
  ~2 ms (norms, rope, attention, 115 launches). TTFA levers only; RTF
  unaffected.
- **Vocoder transformer K=1 GEMMs** at T = 4 and 16 still go through
  `k_conv`; `k_gemm_skinny` needs `EPI_SCALE_RESIDUAL` and `EPI_GELU` to take
  them. Small, but on the TTFA path (the first 4-frame batch).
- **int8 code-predictor weights**: the only lever left on the frame itself
  (2.4 of the 3.3 GB per frame is the code predictor, streamed 15×). A model
  change - not bit-exact, needs the listen test and a parity number. M4.
- **On-device frame loop** (host tail ~0.1 ms/frame, TTFA jitter): M4.
- The model's own no-EOS behaviour on one-token prompts (~15 % of seeds run
  to the 600-frame cap) is bounded, not fixed; a text-derived cap is a
  DESIGN §2 decision.

## 1.7B-Base on gfx1151 (2026-10-13, untuned)

Added at `648a1f2` and measured, not tuned: `MV_ROWS`, the grid, the
block size and the spin cap are all still the values the 0.6B was tuned to. Medians
of 10 `navi-tts bench` runs, bench text at seed 2:

| | 0.6B | 1.7B | ratio |
|---|---|---|---|
| frame | 19.77 ms | 32.65 ms | 1.65× |
| prefill | 16.1 ms | 50.3 ms | 3.12× |
| vocoder | 1.33 ms/frame | 1.34 ms/frame | 1.01× |
| TTFA | 95 ms | 171 ms | 1.80× |
| RTF | 0.267 | 0.433 | 1.62× |

- **The frame is sublinear in the talker.** The talker's weights double; the frame
  costs 1.65×. The code predictor is identical between the two models and is streamed
  15× per frame, so it is a large fixed share that the 1.7B does not pay more for.
  The `small_to_mtp_projection` the 1.7B adds is 16 matvecs of
  1024×2048 f16 per frame, ~64 MB, and it adds no grid barrier.
- **Prefill triples** - 3.12× for 2× the weights, the one place that is worse than
  linear. It is off the per-frame path but it is most of the extra TTFA. Untouched by
  the 0.6B's GEMM tuning; the obvious first place to look if TTFA matters.
- **The vocoder is unchanged**, as it must be: the codec is the same weights in both
  blobs. A 1.01× is a useful sanity check that nothing model-sized leaked into it.
- **Occupancy.** The 1.7B's largest LDS vector is its talker ff (6144 floats) against
  the 0.6B's 4096, so the block needs 50304 B instead of 33920 and goes from 3
  resident blocks per MP to 2. The grid is `multiProcessorCount` × 1 either way,
  so this changes nothing today; it would matter to a `blocks_per_mp` > 2
  experiment, which DESIGN §2 argues against anyway.

### Measurement regime on this box, worth knowing before trusting a number

The Strix Halo box is also a desktop (compositor, browsers), and its iGPU drops to
~618 MHz at idle. That gives two regimes, and mixing them is how you get a wrong
number:

- **Within one warm process** (`--repeats 7` in a single invocation), spread is
  0.5-0.9 %. This is the regime for an A/B: the 0.6B measured 19.1646 ms before the
  1.7B work and 19.1689 ms after, +0.02 %.
- **Across process restarts and a longer session**, spread is 4-10 % and drifts
  upward: the same 0.6B measured 19.17 ms early and 20.47 ms an hour later, with the
  1.7B moving by the same proportion in the same window. The 1.7B/0.6B frame ratio
  stayed 1.63× across both regimes - a ratio survives the drift, an absolute number
  does not.

So: compare interleaved, in one window, and treat a single cold run as ±5 %. The
README's generated medians are over 10 runs and therefore sit in the slower regime,
which is the honest number for "what this box does", not the best one.
