# RTX 3090 PyTorch3D ball-query completion

This directory completes the Reviewer #3 common-operator comparison with
PyTorch3D 0.7.9 built from official source commit
`fdaf9bd6fed7977e4c2056e7c77c640781e58fcd`. The build uses PyTorch
2.7.1+cu118 and one otherwise idle RTX 3090.

The formal result contains 486 records: 81 deterministic S3DIS fixed-250k
crops, pre/post modes, and k=24/32/48. Every setting uses 3 warm-ups and 10
recorded repeats. Radii are copied from the matched local Pointcept result and
are the global 90th percentile of exact kth-neighbor distances. The PyTorch3D
call uses `return_nn=False` and `skip_points_outside_cube=True`; CUDA-event
timing includes `ball_query` and output-distance square root while excluding
I/O, crop/voxel preparation, H2D, and exact-reference construction.

`scripts/validate_pytorch3d_ball_query.py` passed the complete identity,
protocol, timing, radius, and GPU checks. The paper-facing full summary is
`analysis/output/rtx3090_ball_query_operators_20260825/summary.md`.
