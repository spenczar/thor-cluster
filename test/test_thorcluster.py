import thor_cluster
import pyarrow as pa
import pytest
import csv

def test_thorcluster():
    x = pa.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], type=pa.float64())
    y = pa.array([4.0, 5.0, 6.0, 4.1, 3.9, 3.8], type=pa.float64())
    have = thor_cluster.find_clusters(x, y, 1.0, 4)
    want = pa.array([0, -1, -1, 0, 0, 0], type=pa.int32())
    assert have == (want)


@pytest.mark.parametrize("n", [100, 1000, 10000, 30000, 50000, 70000])
@pytest.mark.benchmark(group="thorcluster")
def test_thorcluster_benchmark(benchmark, benchmark_data, n):
    data = benchmark_data
    x_array = data["x"][0:n]
    y_array = data["y"][0:n]
    
    benchmark(thor_cluster.find_clusters, xs=x_array, ys=y_array, eps=0.02, min_cluster_size=4)


@pytest.fixture(scope="session")
def benchmark_data():
    datafile = "./testdata/cluster_input.csv"
    with open(datafile, "r") as f:
        reader = csv.reader(f)
        next(reader)
        x = []
        y = []
        dt = []
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            dt.append(float(row[2]))

    x_array = pa.array(x, type=pa.float64())
    y_array = pa.array(y, type=pa.float64())
    dt_array = pa.array(dt, type=pa.float64())
    return {
        "x": x_array,
        "y": y_array,
        "dt": dt_array,
    }
