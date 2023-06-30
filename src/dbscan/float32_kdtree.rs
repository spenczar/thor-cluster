use crate::dbscan::SearchTree;
use crate::points::XYPoint;
use kiddo::distance;
use kiddo::float::kdtree as kfloat;

// Store points in a 2-dimensional KD-tree of 32-bit floats. Use a
// 16-bit unsigned integer as an index.
pub type PointTree = kfloat::KdTree<f32, u16, 2, 32, u16>;

impl SearchTree for PointTree {
    fn from_points(points: &Vec<XYPoint<f64>>) -> Self {
        let mut tree = kfloat::KdTree::with_capacity(points.len());
        for (idx, point) in points.iter().enumerate() {
            tree.add(&[point.x as f32, point.y as f32], idx as u16);
        }
        tree
    }

    fn neighbors(&self, point: &XYPoint<f64>, radius: f64) -> Vec<usize> {
        let eps = radius as f32;
        let neighbors = self.within_unsorted(
            &[point.x as f32, point.y as f32],
            eps,
            &distance::squared_euclidean,
        );
        neighbors.iter().map(|n| n.item as usize).collect()
    }
}
