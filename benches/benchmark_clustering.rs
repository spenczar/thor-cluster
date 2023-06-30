use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde;
use thor_cluster::{hist2d, quantize, XYPoint};

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
    c.bench_function("quantize", |b| b.iter(|| quantize(black_box(&points), 0.1)));
    let quantized = quantize(&points, 0.1);

    let mut group = c.benchmark_group("hist2d");
    for size in [10, 100, 1000, 10000, 20000, 30000, 40000, 50000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
	    let mut quantized_n = quantized.clone();
	    quantized_n.truncate(size);
            b.iter(|| black_box(hist2d(black_box(quantized_n.clone()))));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
