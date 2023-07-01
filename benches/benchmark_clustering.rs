use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde;
use thor_cluster::points::{XYPoint, XYTPoint};
use thor_cluster::{find_clusters, ClusterAlgorithm};
use thor_cluster::gridsearch::cluster_grid_search;

#[derive(Debug, serde::Deserialize)]
struct TestDataRow {
    x: f64,
    y: f64,
    dt: f64,
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

fn load_testdata_dts() -> Vec<XYTPoint<f64>> {
    let mut reader = csv::Reader::from_path("testdata/cluster_input.csv").unwrap();
    let mut points = Vec::new();
    while let Some(result) = reader.deserialize().next() {
        let record: TestDataRow = result.unwrap();
        let point = XYTPoint::new(record.x, record.y, record.dt);
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
                    black_box(&points_n.clone()),
                    0.02,
                    5,
                    &ClusterAlgorithm::Hotspot2D,
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
                    black_box(&points_n.clone()),
                    0.02,
                    4,
                    &ClusterAlgorithm::DBSCAN,
                ))
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("find_clusters_rtree");
    for size in [10, 100, 1000, 10000, 30000, 50000, 70000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut points_n = points.clone();
            points_n.truncate(size);
            b.iter(|| {
                black_box(find_clusters(
                    black_box(&points_n.clone()),
                    0.02,
                    4,
                    &ClusterAlgorithm::DbscanRStar,
                ))
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("gridsearch");
    let points = load_testdata_dts();
    for size in [10, 25, 50, 100, 150, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut points_n = points.clone();
	    let mut vxs = Vec::new();
	    let mut vys = Vec::new();
	    for i in 0..300 {
		vxs.push(((i-150) as f64)/150.0);
		vys.push(((i-150) as f64)/150.0);		
	    }
            points_n.truncate(size);
            b.iter(|| {
                black_box(cluster_grid_search(
                    black_box(&points_n.clone()),
		    vxs.clone(),
		    vys.clone(),
                    ClusterAlgorithm::DbscanRStar,
                    0.02,
                    4,
		    8,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
