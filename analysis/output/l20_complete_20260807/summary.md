# Benchmark summary

Workbook: `benchmark_results.xlsx`

## SemanticKITTI representative operating point (alpha=4, k=24)

| gpu        | dataset       | mode   |   k |   alpha |   samples |   flashknn_total_ms |   flashknn_recall | exact_method   |   exact_total_ms |   speedup_vs_exact |
|:-----------|:--------------|:-------|----:|--------:|----------:|--------------------:|------------------:|:---------------|-----------------:|-------------------:|
| NVIDIA L20 | SemanticKITTI | post   |  24 |       4 |       110 |             2.39884 |          0.978889 | cukd           |          3.08331 |            1.28533 |
| NVIDIA L20 | SemanticKITTI | pre    |  24 |       4 |       110 |             1.94205 |          0.981446 | cukd           |          3.14735 |            1.62063 |

Speedup is total latency of the exact CUDA k-d tree divided by FlashKNN total latency on the same GPU and query mode.

## Query main table

| gpu        | dataset       | scope       | mode   |   k | method      |   alpha |   samples |   support_points |   query_points |   construction_ms |    query_ms |   total_ms |   recall |
|:-----------|:--------------|:------------|:-------|----:|:------------|--------:|----------:|-----------------:|---------------:|------------------:|------------:|-----------:|---------:|
| NVIDIA L20 | S3DIS         | full        | pre    |  32 | cuda_kdtree |     nan |       272 |         257064   |       257064   |          3.26855  |    2.15769  |    5.42624 | 1        |
| NVIDIA L20 | S3DIS         | full        | pre    |  32 | flann_cuda  |     nan |       272 |         257064   |       257064   |          8.14057  |    6.94029  |   15.0809  | 0.999965 |
| NVIDIA L20 | S3DIS         | full        | pre    |  32 | flashknn    |       4 |       272 |         257064   |       257064   |          1.06065  |    1.59572  |    2.65637 | 0.999914 |
| NVIDIA L20 | S3DIS         | full        | pre    |  32 | nanoflann   |     nan |       272 |         257064   |       257064   |         30.8627   |  747.377    |  778.239   | 0.99996  |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.11792  |    0.340498 |    3.45842 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.30642  |  230.748    |  231.054   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        148.75     |    6.25743  |  155.008   | 0.995671 |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.32107  |    0.403883 |    8.72495 | 0.99988  |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52341  |    0.532879 |    2.05629 | 0.999847 |
| NVIDIA L20 | S3DIS         | sample_part | post   |   8 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.4972   |   75.7243   |  105.222   | 0.999821 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.11621  |    0.543149 |    3.65936 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.30712  |  230.785    |  231.092   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        148.303    |    8.74649  |  157.05    | 0.997714 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.31933  |    0.714944 |    9.03427 | 0.999919 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52506  |    0.55093  |    2.07599 | 0.999891 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  16 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.5151   |  118.118    |  147.633   | 0.9999   |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.11754  |    0.783273 |    3.90082 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.313071 |  230.689    |  231.002   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        148.499    |    7.62191  |  156.121   | 0.998444 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.33191  |    1.11396  |    9.44587 | 0.999958 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52683  |    0.643456 |    2.17028 | 0.999916 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  24 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.495    |  158.794    |  188.289   | 0.999948 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.11936  |    1.08286  |    4.20223 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.305889 |  230.63     |  230.936   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        148.539    |    7.56187  |  156.101   | 0.998814 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.23523  |    1.88027  |   10.1155  | 0.999965 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52794  |    0.691974 |    2.21991 | 0.999853 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  32 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.5248   |  198.362    |  227.886   | 0.99996  |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.12044  |    1.61272  |    4.73316 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.304166 |  252.111    |  252.415   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        148.641    |    8.03624  |  156.677   | 0.999091 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.23668  |    2.96584  |   11.2025  | 0.999979 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52968  |    0.904761 |    2.43444 | 0.998958 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  48 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.4882   |  272.485    |  301.973   | 0.999976 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | cuda_kdtree |     nan |        81 |         250000   |        66363.8 |          3.12113  |    2.14672  |    5.26785 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | faiss_flat  |     nan |        81 |         250000   |        66363.8 |          0.310444 |  253.969    |  254.279   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | faiss_ivf   |     nan |        81 |         250000   |        66363.8 |        149.217    |    6.3776   |  155.594   | 0.996097 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | flann_cuda  |     nan |        81 |         250000   |        66363.8 |          8.23533  |    4.2763   |   12.5116  | 0.999984 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | flashknn    |       4 |        81 |         250000   |        66363.8 |          1.52901  |    1.02199  |    2.551   | 0.993738 |
| NVIDIA L20 | S3DIS         | sample_part | post   |  64 | nanoflann   |     nan |        81 |         250000   |        66363.8 |         29.5547   |  345.478    |  375.033   | 0.999982 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.10127  |    0.729363 |    3.83063 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.328419 |  868.239    |  868.568   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        161.347    |   24.556    |  185.903   | 0.995617 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.27419  |    1.29555  |    9.56973 | 0.999826 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | flashknn    |       4 |        81 |         250000   |       250000   |          1.10193  |    0.646195 |    1.74813 | 0.999818 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |   8 | nanoflann   |     nan |        81 |         250000   |       250000   |         47.0095   |  323.943    |  370.953   | 0.999817 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.10134  |    1.33252  |    4.43386 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.325574 |  873.091    |  873.416   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        160.967    |   34.0932   |  195.06    | 0.997696 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.29505  |    2.47949  |   10.7745  | 0.9999   |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | flashknn    |       4 |        81 |         250000   |       250000   |          1.11673  |    0.867091 |    1.98382 | 0.999894 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  16 | nanoflann   |     nan |        81 |         250000   |       250000   |         46.9525   |  499.285    |  546.237   | 0.999898 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.10366  |    2.0524   |    5.15606 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.320172 |  875.193    |  875.513   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        161.64     |   29.0633   |  190.703   | 0.998435 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.29328  |    4.13134  |   12.4246  | 0.999951 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | flashknn    |       4 |        81 |         250000   |       250000   |          1.11969  |    1.27573  |    2.39541 | 0.999931 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  24 | nanoflann   |     nan |        81 |         250000   |       250000   |         46.864    |  668.362    |  715.226   | 0.999946 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.11129  |    2.87792  |    5.98922 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.320793 |  879.096    |  879.417   | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        161.663    |   29.0947   |  190.757   | 0.998809 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.30003  |    6.56789  |   14.8679  | 0.99996  |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | flashknn    |       4 |        81 |         250000   |       250000   |          1.12643  |    1.52842  |    2.65485 | 0.999901 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  32 | nanoflann   |     nan |        81 |         250000   |       250000   |         46.9838   |  842.554    |  889.538   | 0.999959 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.1051   |    4.52701  |    7.63212 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.336438 | 1007.67     | 1008       | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        161.558    |   50.3582   |  211.916   | 0.999124 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.2989   |   10.7424   |   19.0413  | 0.999976 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | flashknn    |       4 |        81 |         250000   |       250000   |          1.14546  |    2.4689   |    3.61436 | 0.99938  |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  48 | nanoflann   |     nan |        81 |         250000   |       250000   |         46.9744   | 1192.03     | 1239.01    | 0.999975 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | cuda_kdtree |     nan |        81 |         250000   |       250000   |          3.10623  |    6.21985  |    9.32609 | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | faiss_flat  |     nan |        81 |         250000   |       250000   |          0.331162 | 1025.54     | 1025.87    | 1        |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | faiss_ivf   |     nan |        81 |         250000   |       250000   |        161.115    |   24.1819   |  185.297   | 0.996342 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | flann_cuda  |     nan |        81 |         250000   |       250000   |          8.31465  |   15.258    |   23.5726  | 0.999982 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | flashknn    |       4 |        81 |         250000   |       250000   |          1.16522  |    2.90488  |    4.07009 | 0.994525 |
| NVIDIA L20 | S3DIS         | sample_part | pre    |  64 | nanoflann   |     nan |        81 |         250000   |       250000   |         46.9557   | 1519.95     | 1566.9     | 0.999982 |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.23367  |    0.464723 |    2.69839 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.157354 |   59.6529   |   59.8103  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         66.0675   |    3.03835  |   69.1058  | 0.995932 |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.60671  |    0.431662 |    8.03837 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.81682  |    0.529338 |    2.34616 | 0.991192 |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.69956  |    0.742646 |    2.44221 | 0.997159 |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.69223  |    3.40895  |    5.10118 | 0.999311 |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.69369  |   23.1087   |   24.8024  | 0.99985  |
| NVIDIA L20 | SemanticKITTI | full        | post   |   8 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7443   |   52.2687   |   63.013   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.23209  |    0.666775 |    2.89886 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.157488 |   59.6809   |   59.8383  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         66.1253   |    3.1069   |   69.2322  | 0.993628 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.60977  |    0.694367 |    8.30414 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.81359  |    0.547702 |    2.36129 | 0.983214 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.7018   |    0.825653 |    2.52745 | 0.993739 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.6962   |    3.90538  |    5.60158 | 0.998154 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.69633  |   26.3725   |   28.0689  | 0.999555 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  16 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7386   |   82.9414   |   93.68    | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.23609  |    0.847217 |    3.08331 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.15909  |   59.7107   |   59.8698  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         66.3849   |    3.20124  |   69.5861  | 0.993893 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.61813  |    0.995005 |    8.61313 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.81599  |    0.58285  |    2.39884 | 0.978889 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.70374  |    0.97347  |    2.67721 | 0.990281 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.69665  |    4.80301  |    6.49966 | 0.996818 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.69955  |   32.4955   |   34.195   | 0.999145 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  24 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7423   |  111.531    |  122.273   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.23927  |    1.02597  |    3.26524 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.158546 |   59.7221   |   59.8806  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         66.3016   |    3.21336  |   69.5149  | 0.992398 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.61606  |    1.56329  |    9.17935 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.82173  |    0.593777 |    2.41551 | 0.976938 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.71398  |    1.03273  |    2.74671 | 0.987644 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.69752  |    4.98176  |    6.67928 | 0.995497 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.7012   |   33.2877   |   34.9889  | 0.99867  |
| NVIDIA L20 | SemanticKITTI | full        | post   |  32 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7433   |  139.62     |  150.363   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.24281  |    1.4006   |    3.64341 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.15805  |   64.8197   |   64.9778  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         67.4645   |    4.49832  |   71.9628  | 0.986534 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.61122  |    2.37016  |    9.98138 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.82382  |    0.703794 |    2.52762 | 0.976183 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.70453  |    1.27796  |    2.98249 | 0.983891 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.69573  |    5.41623  |    7.11196 | 0.993233 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.69942  |   33.1837   |   34.8831  | 0.997679 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  48 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7423   |  193.523    |  204.265   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | cukd        |     nan |       110 |          80409.1 |        49805.3 |          2.24665  |    1.78069  |    4.02734 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | faiss_flat  |     nan |       110 |          80409.1 |        49805.3 |          0.158851 |   65.0326   |   65.1914  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | faiss_ivf   |     nan |       110 |          80409.1 |        49805.3 |         67.3998   |    4.78488  |   72.1847  | 0.984853 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | flann_cuda  |     nan |       110 |          80409.1 |        49805.3 |          7.61398  |    3.3831   |   10.9971  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | flashknn    |       4 |       110 |          80409.1 |        49805.3 |          1.82594  |    0.730444 |    2.55638 | 0.976173 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | flashknn    |       8 |       110 |          80409.1 |        49805.3 |          1.70558  |    1.3979   |    3.10348 | 0.982301 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | flashknn    |      16 |       110 |          80409.1 |        49805.3 |          1.69897  |    5.8056   |    7.50458 | 0.991825 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | flashknn    |      32 |       110 |          80409.1 |        49805.3 |          1.7005   |   34.7439   |   36.4444  | 0.996783 |
| NVIDIA L20 | SemanticKITTI | full        | post   |  64 | nanoflann   |     nan |       110 |          80409.1 |        49805.3 |         10.7502   |  246.379    |  257.129   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.21559  |    0.473902 |    2.68949 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.159952 |   95.5369   |   95.6968  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         67.1665   |    4.88351  |   72.05    | 0.997749 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.60986  |    0.479127 |    8.08899 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.33147  |    0.524302 |    1.85577 | 0.994019 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.21685  |    0.828891 |    2.04574 | 0.998157 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.21115  |    4.33984  |    5.55098 | 0.999554 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.21983  |   27.7677   |   28.9876  | 0.999904 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |   8 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7439   |   82.2642   |   93.0081  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.237    |    0.695737 |    2.93274 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.160379 |   95.558    |   95.7184  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         66.111    |    4.91346  |   71.0244  | 0.995218 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.60399  |    0.840999 |    8.44499 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.33367  |    0.555146 |    1.88882 | 0.986155 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.21831  |    0.929256 |    2.14756 | 0.995701 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.20999  |    4.90763  |    6.11762 | 0.998792 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.21294  |   31.2567   |   32.4697  | 0.999712 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  16 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7553   |  130.615    |  141.371   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.23924  |    0.908106 |    3.14735 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.161016 |   95.536    |   95.697   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         66.0722   |    4.9869   |   71.0591  | 0.994901 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.59715  |    1.33606  |    8.93321 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.33189  |    0.61016  |    1.94205 | 0.981446 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.21526  |    1.12394  |    2.3392  | 0.992454 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.20942  |    5.92425  |    7.13368 | 0.997881 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.21564  |   37.2935   |   38.5091  | 0.999443 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  24 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7374   |  175.752    |  186.489   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.24567  |    1.11613  |    3.3618  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.159991 |   95.5784   |   95.7384  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         66.0885   |    5.0098   |   71.0983  | 0.992673 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.60208  |    2.26226  |    9.86434 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.33522  |    0.625704 |    1.96093 | 0.978879 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.21844  |    1.19122  |    2.40965 | 0.989898 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.20929  |    6.1152   |    7.32449 | 0.99689  |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.21378  |   37.9998   |   39.2136  | 0.99913  |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  32 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7382   |  219.511    |  230.249   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.24683  |    1.5856   |    3.83243 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.160803 |  104.073    |  104.233   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         66.0538   |    7.09512  |   73.1489  | 0.986933 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.59775  |    3.60508  |   11.2028  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.33494  |    0.869837 |    2.20478 | 0.976547 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.21684  |    1.60041  |    2.81725 | 0.986099 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.21171  |    6.87377  |    8.08548 | 0.995027 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.2148   |   39.6608   |   40.8756  | 0.998468 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  48 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7385   |  304.432    |  315.171   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | cukd        |     nan |       110 |          80409.1 |        80409.1 |          2.24979  |    2.03297  |    4.28276 | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | faiss_flat  |     nan |       110 |          80409.1 |        80409.1 |          0.161043 |  104.35     |  104.511   | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | faiss_ivf   |     nan |       110 |          80409.1 |        80409.1 |         66.6086   |    7.37327  |   73.9819  | 0.983127 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | flann_cuda  |     nan |       110 |          80409.1 |        80409.1 |          7.60299  |    5.18649  |   12.7895  | 1        |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | flashknn    |       4 |       110 |          80409.1 |        80409.1 |          1.34035  |    0.931851 |    2.2722  | 0.974491 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | flashknn    |       8 |       110 |          80409.1 |        80409.1 |          1.22171  |    1.77379  |    2.9955  | 0.984365 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | flashknn    |      16 |       110 |          80409.1 |        80409.1 |          1.21313  |    7.3346   |    8.54773 | 0.993807 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | flashknn    |      32 |       110 |          80409.1 |        80409.1 |          1.21616  |   41.2611   |   42.4773  | 0.997847 |
| NVIDIA L20 | SemanticKITTI | full        | pre    |  64 | nanoflann   |     nan |       110 |          80409.1 |        80409.1 |         10.7492   |  388.112    |  398.861   | 1        |

## Network latency

| gpu        | dataset       | model       | backend    |   samples |   points |   preprocessing_ms |   network_ms |   end_to_end_ms |
|:-----------|:--------------|:------------|:-----------|----------:|---------:|-------------------:|-------------:|----------------:|
| NVIDIA L20 | S3DIS         | DeLA        | cpu_kdtree |        68 |  78366.5 |           761.524  |      16.8497 |        778.373  |
| NVIDIA L20 | S3DIS         | DeLA        | flashknn   |        68 |  78371.6 |            20.0195 |      17.2584 |         37.2779 |
| NVIDIA L20 | S3DIS         | minkunet34c | native     |        68 |  78458.1 |             0      |     103.48   |        103.48   |
| NVIDIA L20 | S3DIS         | octformer   | native     |        68 |  78458.1 |             0      |      86.8038 |         86.8038 |
| NVIDIA L20 | S3DIS         | ptv3        | native     |        68 |  78458.1 |             0      |     126.01   |        126.01   |
| NVIDIA L20 | S3DIS         | spunet      | native     |        68 |  78458.1 |             0      |      39.7939 |         39.7939 |
| NVIDIA L20 | SemanticKITTI | deepla      | cpu_kdtree |        22 |  84287.1 |           548.357  |      25.0954 |        573.453  |
| NVIDIA L20 | SemanticKITTI | deepla      | flashknn   |        22 |  84287.1 |            16.4234 |      24.7043 |         41.1278 |
| NVIDIA L20 | SemanticKITTI | dela        | cpu_kdtree |        22 |  84287.1 |           550.583  |      18.0153 |        568.598  |
| NVIDIA L20 | SemanticKITTI | dela        | flashknn   |        22 |  84287.1 |            16.4645 |      17.5549 |         34.0194 |
| NVIDIA L20 | SemanticKITTI | minkunet34c | native     |        22 |  84287.1 |             0      |     102.656  |        102.656  |
| NVIDIA L20 | SemanticKITTI | octformer   | native     |        22 |  84287.1 |             0      |     113.929  |        113.929  |
| NVIDIA L20 | SemanticKITTI | ptv3        | native     |        22 |  84287.1 |             0      |     137.299  |        137.299  |
| NVIDIA L20 | SemanticKITTI | spunet      | native     |        22 |  84287.1 |             0      |      44.7265 |         44.7265 |
