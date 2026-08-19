# L20 final-kernel full-room throughput

Historical values are comparison-only and are not used by the rebuilt paper figures.

## Point-count bins

| bin | rooms | mean_support_points | flashknn_query_ms | flashknn_mquery_s | cuda_kdtree_query_ms | cuda_kdtree_mquery_s | query_speedup |
|---|---|---|---|---|---|---|---|
| <=250k | 191 | 161755.1780 | 0.9697 | 162.7248 | 1.5137 | 107.4568 | 1.5486 |
| 250k-500k | 57 | 324380.5263 | 2.0471 | 159.3067 | 2.6549 | 126.1144 | 1.3019 |
| 500k-1m | 19 | 689798.4737 | 4.4320 | 155.5733 | 5.0509 | 137.0213 | 1.1434 |
| 1m-2m | 4 | 1251351 | 8.0102 | 156.8591 | 8.4884 | 147.7793 | 1.0614 |
| >2m | 1 | 2424985 | 15.8872 | 152.6380 | 16.5223 | 146.7703 | 1.0400 |

## Representative rooms

| target_points | room | support_points | flashknn_query_ms | cuda_kdtree_query_ms | query_speedup | old_flashknn_query_ms | old_cuda_kdtree_query_ms | old_query_speedup |
|---|---|---|---|---|---|---|---|---|
| 250000 | Area_4/office_21 | 249534 | 1.5065 | 1.9954 | 1.3245 | 1.5068 | 1.9765 | 1.3117 |
| 500000 | Area_6/openspace_1 | 500963 | 3.2878 | 3.8733 | 1.1781 | 3.2949 | 3.8890 | 1.1803 |
| 1000000 | Area_4/lobby_1 | 1017404 | 6.6670 | 7.0795 | 1.0619 | 6.6655 | 7.0712 | 1.0609 |
| 2425000 | Area_2/auditorium_2 | 2424985 | 15.8872 | 16.5223 | 1.0400 | 15.9060 | 16.4368 | 1.0334 |
