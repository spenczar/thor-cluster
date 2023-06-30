# hotspot2d in rust

Input/output are arrow arrays

```py
import thor_cluster
import pandas as pd
import pyarrow as pa

df = pd.read_csv("~/code/b612/thor/bench/data/one_cluster_iteration.csv")
xs = pa.array(df.x.values)
ys = pa.array(df.y.values)

thor_cluster.find_clusters(xs, ys, eps=0.02, min_sample=4)
```

Approx runtime on M1 macbook:


### `min_sample=4`

| n observations | thor_cluster (ms) | thor (ms) |
| -------------- | ----------------- | --------- |
| 100            | 0.15              | 0.02      |
| 1000           | 1.39              | 0.11      |
| 10000          | 12.7              | 10.6      |
| 30000          | 33.5              | 45.6      |
| 50000          | 57.9              | 111       |
| 70000          | 76.4              | 243       |

### `min_sample=5`

| n observations | thor_cluster (ms) | thor (ms) |
| -------------- | ----------------- | --------- |
| 100            | 0.15              | 0.02      |
| 1000           | 1.38              | 0.11      |
| 10000          | 12.4              | 7.15      |
| 30000          | 33.7              | 24.9      |
| 50000          | 58.4              | 51.2      |
| 70000          | 77.7              | 102       |

thor_cluster is pretty much linear in the size of the dataset, while
thor is quadratic, but faster for small datasets.
