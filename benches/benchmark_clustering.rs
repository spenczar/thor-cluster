use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde;
use thor_cluster::points::XYPoint;
use thor_cluster::{find_clusters, ClusterAlgorithm};

#[derive(Debug, serde::Deserialize)]
struct TestDataRow {
    x: f64,
    y: f64,
    dt: f64,
    obs_ids: i64,
}

fn load_testdata() -> Vec<XYPoint<f64>> {
    let mut reader = csv::Reader::from_path("testdata/cluster_input.csv").unwrap();
    let mut points = Vec::new();
    while let Some(result) = reader.deserialize().next() {
        let record: TestDataRow = result.unwrap();
        let point = XYPoint::new(record.x, record.y);
        points.push(point);
    }
    assert_eq!(points.len(), 70598);
    points
}

fn criterion_benchmark(c: &mut Criterion) {
    let points = load_testdata();

    let mut group = c.benchmark_group("find_clusters_hotspot2d");
    for size in [10, 100, 1000, 10000, 30000, 50000, 70000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut points_n = points.clone();
            points_n.truncate(size);
            b.iter(|| {
                black_box(find_clusters(
                    black_box(points_n.clone()),
                    0.02,
                    5,
                    ClusterAlgorithm::Hotspot2D,
                ))
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("find_clusters_dbscan");
    for size in [10, 100, 1000, 10000, 30000, 50000, 70000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut points_n = points.clone();
            points_n.truncate(size);
            b.iter(|| {
                black_box(find_clusters(
                    black_box(points_n.clone()),
                    0.02,
                    4,
                    ClusterAlgorithm::DBSCAN,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
