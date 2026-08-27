# Curated revision evidence snapshot

This snapshot closes the evidence gap between the revised manuscript and the
previous public experiment commit. It adds only retained paper-facing evidence:

- corrected production-kernel S3DIS query and Pointcept ball-query results;
- SemanticKITTI alpha=8 query, network, and checkpoint-compatibility results;
- the matched PyTorch3D ball-query operator result;
- GMSS and upstream `torch_knnquery` diagnostics;
- the final RTX 3090 Nsight Compute CSV/provenance package; and
- the documented concurrent SemanticKITTI training wall-clock observations.

The corresponding runners, validators, and analysis scripts are included.
Smoke runs, co-tenant probes, superseded kernel outputs, and abandoned radial or
log-spherical coordinate-transform experiments are intentionally excluded.
