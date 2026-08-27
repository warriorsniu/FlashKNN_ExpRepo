# SemanticKITTI concurrent training wall-clock

This directory retains the complete-training wall-clock values used by the
revision. DeLA and DeepLA-24 were trained for 100 epochs with 500 steps per
epoch using either FlashKNN or nanoflann. Seed 47 used two concurrent jobs per
backend launch; seeds 48 and 49 used four jobs launched together. Evaluation
was performed separately and is excluded from the interval.

The CSV records the first and final epoch timestamps, total wall-clock,
per-epoch and per-step time, and peak memory. Comparisons are paired within the
same model, seed, and concurrency regime. These measurements support a benefit
under the recorded concurrent configuration; they are not isolated-GPU or
distributed-training throughput claims.
