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
|----------------|-------------------|-----------|
| 100            | 0.06              | 0.02      |
| 1000           | 0.61              | 0.11      |
| 10000          | 5.12              | 10.6      |
| 30000          | 17.52             | 45.6      |
| 50000          | 25.28             | 111       |
| 70000          | 39.76             | 243       |

### `min_sample=5`

| n observations | thor_cluster (ms) | thor (ms) |
|----------------|-------------------|-----------|
| 100            | 0.05              | 0.02      |
| 1000           | 0.61              | 0.11      |
| 10000          | 5.63              | 7.15      |
| 30000          | 15.77             | 24.9      |
| 50000          | 28.98             | 51.2      |
| 70000          | 40.11             | 102       |

thor_cluster is pretty much linear in the size of the dataset, while
thor is quadratic.
