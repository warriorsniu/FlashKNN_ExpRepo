# S3DIS semantic-boundary evaluation

A point is classified as a semantic-boundary point when fewer than 50% of its exact 24-NN support points have the center point's ground-truth class. The support set includes the query point itself.

Evaluated rooms: 68
Evaluated points: 5327301
Semantic-boundary points: 49144 (0.9225%)

| Subset | Method | Points | Accuracy | mAcc | mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| all | FlashKNN | 5327301 | 0.9156 | 0.7872 | 0.7295 |
| all | ExactKNN | 5327301 | 0.9135 | 0.7898 | 0.7286 |
| semantic_boundary | FlashKNN | 49144 | 0.6537 | 0.4886 | 0.3477 |
| semantic_boundary | ExactKNN | 49144 | 0.6537 | 0.4839 | 0.3465 |
| non_boundary | FlashKNN | 5278157 | 0.9181 | 0.7903 | 0.7339 |
| non_boundary | ExactKNN | 5278157 | 0.9160 | 0.7928 | 0.7329 |

Differences below are FlashKNN minus ExactKNN in percentage points.

| Subset | Accuracy difference | mIoU difference |
| --- | ---: | ---: |
| all | +0.208 | +0.090 |
| semantic_boundary | +0.000 | +0.118 |
| non_boundary | +0.210 | +0.100 |
