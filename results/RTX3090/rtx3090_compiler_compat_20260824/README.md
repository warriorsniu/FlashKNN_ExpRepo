# CUDA compiler compatibility validation

## Final implementation

The generated two-register bitonic network is shared by both toolchains. Only
the in-place `SelectPeer` helper accounts for the observed compiler behavior:

- CUDA 11.x keeps the source-level conditional update. nvcc 11.8 lowers it to
  predicated moves after register allocation.
- CUDA 12.x uses an explicit in-place bit mask because nvcc 12.6/sm_89 retains
  the same conditional source as divergent `BRA`/reconvergence regions.

No `launch_bounds`, `maxrregcount`, architecture-specific kernel, or altered
ptxas optimization level is used.

## Static compilation checks

| Toolkit / target | k=16 registers | k=32 registers | Spill | Sorting comparison branch |
|---|---:|---:|---:|---|
| CUDA 11.8 / sm_86 | 51 | 50 | 0 B | predicated update |
| CUDA 12.6 / sm_89 | 52 | 58 | 0 B | none |

For CUDA 12.6, manual SASS inspection found no comparison-controlled
`BRA`/`BSSY`/`BSYNC` in the generated sorting intervals: approximately
`0x11b0--0x1cb0` for k=16 and `0x1170--0x2190` for k=32. Branches outside
these intervals implement kernel loops, bounds handling, and reconvergence
unrelated to the sorting comparisons. CUDA 12.8 must still be confirmed on
the L20 machine.

Lowering the CUDA 12.6 ptxas optimization level did not resolve the trade-off:
O0/O1/O2/O3 used 74/82, 57/67, 59/65, and 59/65 registers for k=16/k=32,
respectively. `register-usage-level=0` used 56/65 registers. Stage fences,
packed pairs, integer ordering keys, and narrow inline-PTX compare/select
prototypes were rejected because they increased register pressure.

## RTX 3090 protocol

- GPU 0 only; no co-tenant process
- PyTorch 2.7.1+cu118; CUDA toolkit 11.8; sm_86
- 81 S3DIS fixed-250k `sample_part` pre-query rooms
- 5 warm-ups and 20 timed repeats per room
- Summary unit: per-room query mean; SD and 95% CI are across rooms

## Results

| k | Final query mean (ms) | Per-room SD | 95% CI | Mean recall vs cudaKDTree |
|---:|---:|---:|---:|---:|
| 16 | 0.9594 | 0.0422 | [0.9501, 0.9688] | 0.999998 |
| 32 | 1.5881 | 0.0419 | [1.5788, 1.5973] | 0.999986 |

Against the historical correct branchy build (1.0335 ms for k=16 and
1.7101 ms for k=32), the final build is 7.15% and 7.12% faster and wins in
81/81 paired rooms for both k values. The paired cudaKDTree controls changed
only from 3.7429 to 3.7415 ms and from 10.1677 to 10.1604 ms, so the measured
gain is not explained by machine-load drift. This comparison includes the
current compare-split optimization and must not attribute the full gain solely
to the compiler compatibility helper.

`smoke_all_k.json` additionally validates k={8,16,24,32,48,64} on one 250k
room. `formal.json` contains the 81-room measurements.
