use crate::points::{XYPoint, XYTPoint};
use crate::{find_clusters, ClusterAlgorithm};
use std::thread;
use std::sync::mpsc::channel;


pub struct GridSearchResult {
    vx: f64,
    vy: f64,
    cluster_labels: Vec<i32>,
}

fn apply_velocity(vx: f64, vy: f64, points: &Vec<XYTPoint<f64>>) -> Vec<XYPoint<f64>> {
    let mut new_points = Vec::with_capacity(points.len());
    for p in points.iter() {
        let new_point = XYPoint {
            x: p.x + vx * p.t,
            y: p.y + vy * p.t,
        };
        new_points.push(new_point);
    }
    new_points
}

pub fn cluster_grid_search(
    points: &Vec<XYTPoint<f64>>,
    vxs: Vec<f64>,
    vys: Vec<f64>,
    alg: ClusterAlgorithm,
    eps: f64,
    min_cluster_size: usize,
    n_workers: usize,
) -> Vec<GridSearchResult> {
    if n_workers == 1 {
        return cluster_grid_search_serial(points, vxs, vys, alg, eps, min_cluster_size);
    }
    // Partition the work to be done
    let mut vx_chunks = vxs.chunks(n_workers);
    let (tx, rx) = channel();
    let mut handles = Vec::new();
    for i in 0..n_workers {
        let tx = tx.clone();
        let vx_chunk = vx_chunks.next().unwrap().to_vec();
        let vys = vys.clone();
        let points = points.clone();
        let alg = alg.clone();
        let eps = eps.clone();
        let min_cluster_size = min_cluster_size.clone();
        handles.push(thread::spawn(move || {
            for vx in vx_chunk.into_iter() {
                for vy in &vys {
                    let xy_points = apply_velocity(vx, *vy, &points);
                    let cluster_labels = find_clusters(&xy_points, eps, min_cluster_size, &alg);
                    let result = GridSearchResult {
                        vx: vx,
                        vy: *vy,
                        cluster_labels: cluster_labels,
                    };
                    tx.send(result).unwrap();
                }
            }
        }));
    }
    drop(tx);
    let mut results = Vec::new();
    while let Ok(result) = rx.recv() {
        results.push(result);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    results
}

fn cluster_grid_search_serial(
    points: &Vec<XYTPoint<f64>>,
    vxs: Vec<f64>,
    vys: Vec<f64>,
    alg: ClusterAlgorithm,
    eps: f64,
    min_cluster_size: usize,
) -> Vec<GridSearchResult> {
    let mut results = Vec::new();
    for vx in vxs.iter() {
        for vy in vys.iter() {
            let xy_points = apply_velocity(*vx, *vy, &points);
            let cluster_labels = find_clusters(&xy_points, eps, min_cluster_size, &alg);
            let result = GridSearchResult {
                vx: *vx,
                vy: *vy,
                cluster_labels: cluster_labels,
            };
            results.push(result);
        }
    }
    results
}

#[test]
fn test_grid_search() {
    let points = vec![
        XYTPoint { x: 0.0, y: 0.0, t: 0.0 },
        XYTPoint { x: 0.0, y: 0.0, t: 0.0 },
        XYTPoint { x: 0.0, y: 0.0, t: 1.0 },
        XYTPoint { x: 0.4, y: 0.4, t: 1.0 },
        XYTPoint { x: 10.0, y: 10.0, t: 10.0},
        XYTPoint { x: 10.0, y: 10.1, t: 10.0 },
    ];
    let vxs = vec![0.0, -0.5, -1.0];
    let vys = vec![0.0, -0.5, -1.0];
    let results = cluster_grid_search(&points, vxs, vys, ClusterAlgorithm::DBSCAN, 0.5, 4, 1);
    assert_eq!(results.len(), 9);
    assert_eq!(results[0].vx, 0.0);
    assert_eq!(results[0].vy, 0.0);
    assert_eq!(results[0].cluster_labels.len(), 6);
    assert_eq!(results[0].cluster_labels, vec![1, 1, 1, 1, -1, -1]);

    assert_eq!(results[8].vx, -1.0);
    assert_eq!(results[8].vy, -1.0);
    assert_eq!(results[8].cluster_labels.len(), 6);
    assert_eq!(results[8].cluster_labels, vec![1, 1, -1, -1, 1, 1]);
}
