use fixed::types::extra::U14;
use fixed::FixedU16;
use kiddo::fixed::distance as kfixed_distance;
use kiddo::fixed::kdtree as kfixed;

use crate::dbscan::SearchTree;
use crate::points::XYPoint;

pub type FixedPointTree = kfixed::KdTree<FixedU16<U14>, u32, 2, 32, u32>;
impl SearchTree for FixedPointTree {
    fn from_points(points: &Vec<XYPoint<f64>>) -> Self {
        let mut tree = kfixed::KdTree::with_capacity(points.len());
        for (idx, point) in points.iter().enumerate() {
            tree.add(&to_fixed_point(point), idx as u32);
        }
        tree
    }

    fn neighbors(&self, point: &XYPoint<f64>, radius: f64) -> Vec<usize> {
        let eps = FixedU16::<U14>::from_num(radius);
        let neighbors = self.within_unsorted(
            &to_fixed_point(point),
            eps,
            &kfixed_distance::squared_euclidean,
        );
        neighbors.iter().map(|n| n.item as usize).collect()
    }
}

#[inline]
fn to_fixed_point(p: &XYPoint<f64>) -> [FixedU16<U14>; 2] {
    [
        FixedU16::<U14>::from_num(p.x),
        FixedU16::<U14>::from_num(p.y),
    ]
}
