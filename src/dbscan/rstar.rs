use rstar::primitives::GeomWithData;
use rstar::RTree;

use crate::dbscan::SearchTree;
use crate::points::XYPoint;

pub type Tree = RTree<TreeEntry>;

impl SearchTree for Tree {
    fn from_points(points: &Vec<XYPoint<f64>>) -> Self {
        let points: Vec<TreeEntry> = points
            .iter()
            .enumerate()
            .map(|(i, p)| point_to_tree_entry(p, i as u16))
            .collect();
        Tree::bulk_load(points)
    }

    fn neighbors(&self, point: &XYPoint<f64>, radius: f64) -> Vec<usize> {
        self.locate_within_distance([point.x, point.y], radius)
            .map(|p| p.data as usize)
            .collect()
    }
}

type TreeEntry = GeomWithData<[f64; 2], u16>;

fn point_to_tree_entry(point: &XYPoint<f64>, idx: u16) -> TreeEntry {
    GeomWithData::new([point.x, point.y], idx)
}
