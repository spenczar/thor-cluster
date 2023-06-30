use crate::points::XYPoint;
use kiddo::distance;
use kiddo::float::kdtree;

// Store points in a 2-dimensional KD-tree of 32-bit floats. Use a
// 16-bit unsigned integer as an index.
type PointTree = kdtree::KdTree<f32, u16, 2, 32, u16>;

// TODO: try a fixed-point tree

pub fn find_clusters_dbscan(
    points: Vec<XYPoint<f64>>,
    eps: f64,
    min_cluster_size: usize,
) -> Vec<i32> {
    // First, build a kd-tree of the points
    let mut tree: PointTree = kdtree::KdTree::with_capacity(points.len());
    points.iter().enumerate().for_each(|(i, p)| {
        tree.add(&[p.x as f32, p.y as f32], i as u16);
    });

    dbscan(&points, tree, eps as f32, min_cluster_size)
}

fn dbscan(
    points: &Vec<XYPoint<f64>>,
    tree: PointTree,
    eps: f32,
    min_cluster_size: usize,
) -> Vec<i32> {
    let mut labels: Vec<DBScanClassification> = vec![DBScanClassification::Undefined; points.len()];
    let mut cluster_idx: u16 = 0;

    for (i, point) in points.iter().enumerate() {
        if labels[i] != DBScanClassification::Undefined {
            // Already visited
            continue;
        }
        let neighbors = tree.within_unsorted(
            &[point.x as f32, point.y as f32],
            eps,
            &distance::squared_euclidean,
        );

        if neighbors.len() < min_cluster_size {
            // Too small
            labels[i] = DBScanClassification::Noise;
            continue;
        }
        // Big enough, hooray!
        cluster_idx += 1;
        labels[i] = DBScanClassification::Core(cluster_idx);
        let mut queue = neighbors;
        while let Some(neighbor) = queue.pop() {
            let neighbor_idx = neighbor.item as usize;
            let neighbor_label = &labels[neighbor_idx];
            if *neighbor_label == DBScanClassification::Noise {
                // Maybe you can join our cluster?
                labels[neighbor_idx] = DBScanClassification::Border(cluster_idx);
            } else if *neighbor_label != DBScanClassification::Undefined {
                // You're already with someone else
                continue;
            }
            let neighbor = &points[neighbor_idx];
            let neighbors_of_neighbor = tree.within_unsorted(
                &[neighbor.x as f32, neighbor.y as f32],
                eps,
                &distance::squared_euclidean,
            );
            if neighbors_of_neighbor.len() >= min_cluster_size {
                // You're big enough to join us
                labels[neighbor_idx] = DBScanClassification::Core(cluster_idx);
                queue.extend(neighbors_of_neighbor);
            }
        }
    }

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

#[derive(Debug, Clone, PartialEq)]
enum DBScanClassification {
    Undefined,
    Noise,
    Border(u16),
    Core(u16),
}
