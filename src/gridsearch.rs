use crate::points::{XYPoint, XYTPoint};
use crate::{find_clusters, ClusterAlgorithm};
use std::sync::mpsc::channel;
use std::thread;

pub struct GridSearchResult {
    pub vx: f64,
    pub vy: f64,
    pub cluster_labels: Vec<i32>,
}

fn apply_velocity(vx: f64, vy: f64, points: &Vec<XYTPoint<f64>>) -> Vec<XYPoint<f64>> {
    let mut new_points = Vec::with_capacity(points.len());
    for p in points.iter() {
        let new_point = XYPoint {
            x: p.x - vx * p.t,
            y: p.y - vy * p.t,
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
    n_threads: usize,
) -> Vec<GridSearchResult> {
    if n_threads == 1 {
        return cluster_grid_search_serial(points, vxs, vys, alg, eps, min_cluster_size);
    }
    let mut vx_chunks = vxs.chunks(work_chunk_size(n_threads, vxs.len()));
    let (tx, rx) = channel();
    let mut handles = Vec::new();
    for _i in 1..n_threads {
        let tx = tx.clone();
        let vx_chunk = vx_chunks.next();
        if vx_chunk.is_none() {
            // We've run out of chunks to process
            break;
        }
        let vx_chunk = vx_chunk.unwrap().to_vec();
        let vys = vys.clone();
        let points = points.clone();
        let alg = alg.clone();
        let eps = eps.clone();
        let min_cluster_size = min_cluster_size.clone();

        let thread_name = format!("grid_search_{}", _i);
        let threadbuilder = thread::Builder::new().name(thread_name);

        let core_ids = core_affinity::get_core_ids().unwrap();
        let core_id = core_ids[_i % core_ids.len()];
        let handle = threadbuilder.spawn(move || {
            core_affinity::set_for_current(core_id);
            for vx in vx_chunk.iter() {
                for vy in &vys {
                    let xy_points = apply_velocity(*vx, *vy, &points);
                    let cluster_labels = find_clusters(&xy_points, eps, min_cluster_size, &alg);
                    let result = GridSearchResult {
                        vx: *vx,
                        vy: *vy,
                        cluster_labels: cluster_labels,
                    };
                    tx.send(result).unwrap();
                }
            }
        });
        handles.push(handle.unwrap());
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

fn work_chunk_size(n_threads: usize, n_vxs: usize) -> usize {
    let chunk_size = n_vxs / n_threads;
    if n_vxs % n_threads == 0 {
        chunk_size
    } else {
        chunk_size + 1
    }
}

#[test]
fn test_work_chunk_size() {
    // Partition the work to be done
    // for 2 threads
    // 10 -> 5, 5
    // 11 -> 6, 5
    // 12 -> 6, 6
    // 13 -> 7, 6
    assert_eq!(work_chunk_size(2, 10), 5);
    assert_eq!(work_chunk_size(2, 11), 6);
    assert_eq!(work_chunk_size(2, 12), 6);
    assert_eq!(work_chunk_size(2, 13), 7);
    // for 3 threads
    // 10 -> 4, 4, 2
    // 11 -> 4, 4, 3
    // 12 -> 4, 4, 4
    // 13 -> 5, 5, 3
    // 14 -> 5, 5, 4
    // 15 -> 5, 5, 5
    // 16 -> 6, 5, 5
    assert_eq!(work_chunk_size(3, 10), 4);
    assert_eq!(work_chunk_size(3, 11), 4);
    assert_eq!(work_chunk_size(3, 12), 4);
    assert_eq!(work_chunk_size(3, 13), 5);
    assert_eq!(work_chunk_size(3, 14), 5);
    assert_eq!(work_chunk_size(3, 15), 5);
    assert_eq!(work_chunk_size(3, 16), 6);
}

#[test]
fn test_grid_search() {
    let points = vec![
        XYTPoint {
            x: 0.0,
            y: 0.0,
            t: 0.0,
        },
        XYTPoint {
            x: 0.0,
            y: 0.0,
            t: 0.0,
        },
        XYTPoint {
            x: 0.0,
            y: 0.0,
            t: 1.0,
        },
        XYTPoint {
            x: 0.4,
            y: 0.4,
            t: 1.0,
        },
        XYTPoint {
            x: 10.0,
            y: 10.0,
            t: 10.0,
        },
        XYTPoint {
            x: 10.0,
            y: 10.1,
            t: 10.0,
        },
    ];
    let vxs = vec![0.0, 0.5, 1.0];
    let vys = vec![0.0, 0.5, 1.0];
    let results = cluster_grid_search(&points, vxs, vys, ClusterAlgorithm::DBSCAN, 0.5, 4, 1);
    assert_eq!(results.len(), 9);
    assert_eq!(results[0].vx, 0.0);
    assert_eq!(results[0].vy, 0.0);
    assert_eq!(results[0].cluster_labels.len(), 6);
    assert_eq!(results[0].cluster_labels, vec![1, 1, 1, 1, -1, -1]);

    assert_eq!(results[8].vx, 1.0);
    assert_eq!(results[8].vy, 1.0);
    assert_eq!(results[8].cluster_labels.len(), 6);
    assert_eq!(results[8].cluster_labels, vec![1, 1, -1, -1, 1, 1]);
}

#[test]
fn test_grid_search_multithreaded() {
    let points = vec![
        // Make a cluster at x=0, y=0 that only appears at velocity
        // vx=0, vy=0.
        XYTPoint {
            x: 0.0,
            y: 0.0,
            t: 0.0,
        },
        XYTPoint {
            x: 0.2,
            y: 0.0,
            t: 4.0,
        },
        XYTPoint {
            x: 0.0,
            y: 0.3,
            t: 8.0,
        },
        XYTPoint {
            x: 0.1,
            y: 0.2,
            t: 3.0,
        },
        // Make a cluster at x=10, y=10 that only appears at velocity vx=1, vy=0.
        XYTPoint {
            x: 10.0,
            y: 10.0,
            t: 0.0,
        },
        XYTPoint {
            x: 11.2,
            y: 10.0,
            t: 1.0,
        },
        XYTPoint {
            x: 12.0,
            y: 10.3,
            t: 2.0,
        },
        XYTPoint {
            x: 15.1,
            y: 10.2,
            t: 5.0,
        },
        // Make a cluster at x=5, y=5 that only appears at velocity vx=-1, vy=1.
        XYTPoint {
            x: 5.0,
            y: 2.0,
            t: 0.0,
        },
        XYTPoint {
            x: 4.2,
            y: 3.0,
            t: 1.0,
        },
        XYTPoint {
            x: 3.0,
            y: 4.3,
            t: 2.0,
        },
        XYTPoint {
            x: 2.1,
            y: 5.2,
            t: 3.0,
        },
        // Make some noise.
        XYTPoint {
            x: 0.0,
            y: 3.0,
            t: 7.0,
        },
        XYTPoint {
            x: 2.0,
            y: 4.1,
            t: 2.0,
        },
        XYTPoint {
            x: 3.0,
            y: 5.3,
            t: 3.0,
        },
        XYTPoint {
            x: 4.1,
            y: 6.2,
            t: 4.0,
        },
    ];
    let vxs = vec![-1.0, 0.0, 1.0];
    let vys = vec![-1.0, 0.0, 1.0];

    let n_points = points.len();
    let n_vxvy_pairs = vxs.len() * vys.len();
    // Find clusters using 4 threads (1 main, 3 workers).
    let results = cluster_grid_search(&points, vxs, vys, ClusterAlgorithm::DBSCAN, 0.5, 4, 4);

    assert_eq!(
        results.len(),
        n_vxvy_pairs,
        "there should be one result for every vx, vy pair"
    );

    // We expect clusters for vx=0, vy=0, vx=1, vy=0, and vx=-1, vy=1.
    for result in results {
        // In each case, the length of result.cluster_labels should equal the length of points.
        assert_eq!(
            result.cluster_labels.len(),
            n_points,
            "there should be one cluster label for every point; missing for vx={}, vy={}",
            result.vx,
            result.vy
        );

        // Match on vx, vy to check the cluster labels.
        match result {
            result if result.vx == 0.0 && result.vy == 0.0 => {
                assert_eq!(
                    result.cluster_labels[0..4],
                    vec![1, 1, 1, 1],
                    "cluster labels for vx={}, vy={} are incorrect: missing cluster",
                    result.vx,
                    result.vy
                );
                for label in result.cluster_labels[4..].iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }
            }
            result if result.vx == 1.0 && result.vy == 0.0 => {
                assert_eq!(
                    result.cluster_labels[4..8],
                    vec![1, 1, 1, 1],
                    "cluster labels for vx={}, vy={} are incorrect: missing cluster",
                    result.vx,
                    result.vy
                );
                for label in result.cluster_labels[0..4].iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }

                for label in result.cluster_labels[8..].iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }
            }
            result if result.vx == -1.0 && result.vy == 1.0 => {
                assert_eq!(
                    result.cluster_labels[8..12],
                    vec![1, 1, 1, 1],
                    "cluster labels for vx={}, vy={} are incorrect: missing cluster",
                    result.vx,
                    result.vy
                );
                for label in result.cluster_labels[0..8].iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }

                for label in result.cluster_labels[12..].iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }
            }
            _ => {
                for label in result.cluster_labels.iter() {
                    assert_eq!(*label, -1, "cluster labels for vx={}, vy={} are incorrect: spurious cluster in labels: {:?}", result.vx, result.vy, result.cluster_labels);
                }
            }
        };
    }
}
