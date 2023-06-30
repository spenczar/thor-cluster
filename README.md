# hotspot2d in rust

## Not tested enough to trust!!!b

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
| 100            | 0.02              | 0.02      |
| 1000           | 0.13              | 0.11      |
| 10000          | 1.15              | 10.6      |
| 30000          | 3.09              | 45.6      |
| 50000          | 5.61              | 111       |
| 70000          | 7.44              | 243       |

### `min_sample=5`

| n observations | thor_cluster (ms) | thor (ms) |
|----------------|-------------------|-----------|
| 100            | 0.02              | 0.02      |
| 1000           | 0.11              | 0.11      |
| 10000          | 1.18              | 7.15      |
| 30000          | 3.19              | 24.9      |
| 50000          | 5.84              | 51.2      |
| 70000          | 7.38              | 102       |

thor_cluster is pretty much linear in the size of the dataset, while
thor is quadratic.
