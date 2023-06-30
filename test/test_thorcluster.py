import thor_cluster
import pyarrow as pa

def test_thorcluster():
    x = pa.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], type=pa.float64())
    y = pa.array([4.0, 5.0, 6.0, 4.1, 3.9, 3.8], type=pa.float64())
    have = thor_cluster.find_clusters(x, y, 1.0, 4)
    want = [[0, 3, 4, 5]]
    assert have == (want)
