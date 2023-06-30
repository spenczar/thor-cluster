pub mod fixed16_kdtree;
pub mod float32_kdtree;
pub mod rstar;
use crate::points::XYPoint;

#[derive(Debug, Clone, PartialEq)]
enum DBScanClassification {
    Undefined,
    Noise,
    Border(u16),
    Core(u16),
}

pub trait SearchTree {
    fn from_points(points: &Vec<XYPoint<f64>>) -> Self;
    fn neighbors(&self, point: &XYPoint<f64>, radius: f64) -> Vec<usize>;
}

pub fn find_clusters<T: SearchTree>(
    points: Vec<XYPoint<f64>>,
    eps: f64,
    min_cluster_size: usize,
) -> Vec<i32> {
    let tree: T = T::from_points(&points);

    let labels = dbscan(&points, &tree, eps, min_cluster_size);
    labels
        .iter()
        .map(|label| match label {
            DBScanClassification::Noise => -1,
            DBScanClassification::Border(i) => *i as i32,
            DBScanClassification::Core(i) => *i as i32,
            DBScanClassification::Undefined => -1,
        })
        .collect()
}

fn dbscan(
    points: &Vec<XYPoint<f64>>,
    tree: &impl SearchTree,
    eps: f64,
    min_cluster_size: usize,
) -> Vec<DBScanClassification> {
    let mut labels: Vec<DBScanClassification> = vec![DBScanClassification::Undefined; points.len()];
    let mut cluster_idx: u16 = 0;

    for (i, point) in points.iter().enumerate() {
        if labels[i] != DBScanClassification::Undefined {
            // Already visited
            continue;
        }
        let neighbors = tree.neighbors(point, eps);

        if neighbors.len() < min_cluster_size {
            // Too small
            labels[i] = DBScanClassification::Noise;
            continue;
        }
        // Big enough, hooray!
        cluster_idx += 1;
        labels[i] = DBScanClassification::Core(cluster_idx);
        let mut queue = neighbors;
        while let Some(neighbor_idx) = queue.pop() {
            let neighbor_label = &labels[neighbor_idx];
            if *neighbor_label == DBScanClassification::Noise {
                // Maybe you can join our cluster?
                labels[neighbor_idx] = DBScanClassification::Border(cluster_idx);
            } else if *neighbor_label != DBScanClassification::Undefined {
                // You're already with someone else
                continue;
            }
            let _neighbor = &points[neighbor_idx];
            let neighbors_of_neighbor = tree.neighbors(point, eps);
            if neighbors_of_neighbor.len() >= min_cluster_size {
                // You're big enough to join us
                labels[neighbor_idx] = DBScanClassification::Core(cluster_idx);
                queue.extend(neighbors_of_neighbor);
            }
        }
    }
    labels
}
