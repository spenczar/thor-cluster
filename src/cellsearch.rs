use log::debug;

use std::collections::HashMap;

use kiddo::float::distance::manhattan;
use kiddo::float::kdtree;
use ordered_float::OrderedFloat;

use crate::points::{XYPoint, XYTPoint};

type XYPoint32 = XYPoint<f32>;
type XYTPoint32 = XYTPoint<f32>;

pub struct ThorCell {
    subtrees: Vec<ThorSubtree>,
    dts: HashMap<OrderedFloat<f32>, usize>,
    sorted_dts: Vec<OrderedFloat<f32>>,
}

fn sorted_insert(vec: &mut Vec<OrderedFloat<f32>>, val: f32) {
    let val = OrderedFloat(val);
    let idx = vec.partition_point(|&x| x < val);
    vec.insert(idx, val);
}

impl ThorCell {
    pub fn new() -> ThorCell {
        ThorCell {
            subtrees: Vec::new(),
            dts: HashMap::new(),
            sorted_dts: Vec::new(),
        }
    }

    pub fn add_point(&mut self, dt: f32, point: XYPoint32) {
        match self.dts.get(&OrderedFloat(dt)) {
            Some(subtree_idx) => {
                self.subtrees[*subtree_idx].add_point(point);
            }
            None => {
                self.dts.insert(OrderedFloat(dt), self.subtrees.len());
                sorted_insert(&mut self.sorted_dts, dt);
                self.subtrees.push(ThorSubtree::new(dt, vec![point]));
            }
        }
    }

    pub fn add_points(&mut self, dt: f32, points: Vec<XYPoint32>) {
        match self.dts.get(&OrderedFloat(dt)) {
            Some(subtree_idx) => {
                self.subtrees[*subtree_idx].add_points(points);
            }
            None => {
                self.sorted_dts.push(OrderedFloat(dt));
                sorted_insert(&mut self.sorted_dts, dt);
                self.subtrees.push(ThorSubtree::new(dt, points));
            }
        }
    }

    pub fn find_clusters2(
        &self,
        eps: f32,
        min_weight: usize,
        vx: f32,
        vy: f32,
    ) -> Vec<Vec<XYTPoint32>> {
        // The idea is to find all points that are within eps, but only in *later* subtrees.

        // Labels for each point in each subtree
        let mut labels: Vec<Vec<ClusterClassification>> = self
            .sorted_dts
            .iter()
            .map(|dt| &self.subtrees[self.dts[dt]])
            .map(|subtree| vec![ClusterClassification::Undefined; subtree.points.len()])
            .collect();

        let mut cluster_idx: usize = 0;

        // Go in DT order
        for (i, dt) in self.sorted_dts.iter().enumerate() {
            let subtree = &self.subtrees[self.dts[&dt]];
            for (j, point) in subtree.points.iter().enumerate() {
                if labels[i][j] != ClusterClassification::Undefined {
                    // Already visited
                    continue;
                }

                let mut possible_cluster_points = Vec::new();

                let followers = self.sorted_dts[i..]
                    .iter()
                    .enumerate()
                    .map(|(k, dt)| (k + i, &self.subtrees[self.dts[dt]]));
                for (k, follower_subtree) in followers {
                    // Modify point to adjust by velocity
                    let point = XYPoint32 {
                        x: point.x + vx * (follower_subtree.dt),
                        y: point.y + vy * (follower_subtree.dt),
                    };

                    if let Some(idx) = follower_subtree.nearest_within(&point, eps) {
                        possible_cluster_points.push((k, idx));
                        labels[k][idx] = ClusterClassification::Border(cluster_idx);
                    }
                }

                if possible_cluster_points.len() < min_weight {
                    // Too small
                    labels[i][j] = ClusterClassification::Noise;
                    for (k, idx) in possible_cluster_points {
                        labels[k][idx] = ClusterClassification::Noise;
                    }
                } else {
                    // New cluster
                    cluster_idx += 1;
                    labels[i][j] = ClusterClassification::Core(cluster_idx);
                    for (k, idx) in possible_cluster_points {
                        labels[k][idx] = ClusterClassification::Core(cluster_idx);
                    }
                }
            }
        }
        // All points are labeled. Now organize the results.
        let mut clusters: Vec<Vec<XYTPoint32>> = vec![Vec::new(); cluster_idx];

        for (subtree_idx, subtree_point_classifications) in labels.iter().enumerate() {
            for (point_idx, point_classification) in
                subtree_point_classifications.iter().enumerate()
            {
                if let ClusterClassification::Core(cluster_idx)
                | ClusterClassification::Border(cluster_idx) = point_classification
                {
                    clusters[*cluster_idx - 1].push(XYTPoint32 {
                        x: self.subtrees[subtree_idx].points[point_idx].x,
                        y: self.subtrees[subtree_idx].points[point_idx].y,
                        t: self.subtrees[subtree_idx].dt,
                    });
                }
            }
        }

        clusters
    }

    pub fn find_clusters(
        &self,
        eps: f32,
        min_weight: usize,
        vx: f32,
        vy: f32,
    ) -> Vec<Vec<XYTPoint32>> {
        // Labels for each point in each subtree
        let mut labels: Vec<Vec<ClusterClassification>> = self
            .subtrees
            .iter()
            .map(|subtree| vec![ClusterClassification::Undefined; subtree.points.len()])
            .collect();

        let mut cluster_idx: usize = 0;

        for (i, subtree) in self.subtrees.iter().enumerate() {
            for (j, point) in subtree.points.iter().enumerate() {
                if labels[i][j] != ClusterClassification::Undefined {
                    // Already visited
                    continue;
                }

                // Gather neighbors from *all* subtrees
                let mut neighbors = self.neighbors(&point, eps, vx, vy);
                if neighbors.len() < min_weight {
                    // Too small
                    labels[i][j] = ClusterClassification::Noise;
                    continue;
                }

                // New cluster
                cluster_idx += 1;
                labels[i][j] = ClusterClassification::Core(cluster_idx);

                while let Some(neighbor_idx) = neighbors.pop() {
                    let label = &mut labels[neighbor_idx.subtree_idx][neighbor_idx.point_idx];

                    if *label == ClusterClassification::Noise {
                        // You aren't a cluster on your own, but you're big enough to join us
                        *label = ClusterClassification::Border(cluster_idx);
                        continue;
                    }
                    if *label != ClusterClassification::Undefined {
                        // Already claimed
                        continue;
                    }
                    // You're a new core member maybe
                    let neighbor_point =
                        &self.subtrees[neighbor_idx.subtree_idx].points[neighbor_idx.point_idx];
                    let neighbors_of_neighbor = self.neighbors(&neighbor_point, eps, vx, vy);
                    if neighbors_of_neighbor.len() >= min_weight {
                        // Join our cluster
                        *label = ClusterClassification::Core(cluster_idx);
                        neighbors.extend(neighbors_of_neighbor);
                    }
                }
            }
        }

        // All points are labeled. Now organize the results.
        let mut clusters: Vec<Vec<XYTPoint32>> = vec![Vec::new(); cluster_idx];

        for (subtree_idx, subtree_point_classifications) in labels.iter().enumerate() {
            for (point_idx, point_classification) in
                subtree_point_classifications.iter().enumerate()
            {
                if let ClusterClassification::Core(cluster_idx)
                | ClusterClassification::Border(cluster_idx) = point_classification
                {
                    clusters[*cluster_idx - 1].push(XYTPoint32 {
                        x: self.subtrees[subtree_idx].points[point_idx].x,
                        y: self.subtrees[subtree_idx].points[point_idx].y,
                        t: self.subtrees[subtree_idx].dt,
                    });
                }
            }
        }

        clusters
    }

    fn neighbors(&self, point: &XYPoint32, eps: f32, vx: f32, vy: f32) -> Vec<SubtreeNeighbor> {
        let mut neighbors = Vec::new();
        for (subtree_idx, subtree) in self.subtrees.iter().enumerate() {
            let mut point = point.clone();
            point.x += vx * subtree.dt;
            point.y += vy * subtree.dt;
            for neighbor_idx in subtree.neighbor_indexes(&point, eps) {
                neighbors.push(SubtreeNeighbor {
                    subtree_idx: subtree_idx,
                    point_idx: neighbor_idx,
                });
            }
        }
        neighbors
    }
}

struct SubtreeNeighbor {
    pub subtree_idx: usize,
    pub point_idx: usize,
}

struct ThorSubtree {
    pub point_index: kdtree::KdTree<f32, usize, 2, 32, u32>,
    pub points: Vec<XYPoint32>,
    pub dt: f32,
}

impl ThorSubtree {
    pub fn new(dt: f32, points: Vec<XYPoint32>) -> ThorSubtree {
        let mut point_tree = kdtree::KdTree::with_capacity(points.len());
        for (i, p) in points.iter().enumerate() {
            point_tree.add(&[p.x, p.y], i);
        }
        ThorSubtree {
            point_index: point_tree,
            points: points,
            dt: dt,
        }
    }

    pub fn add_point(&mut self, point: XYPoint32) {
        self.point_index.add(&[point.x, point.y], self.points.len());
        self.points.push(point);
    }

    pub fn add_points(&mut self, points: Vec<XYPoint32>) {
        for point in points {
            self.add_point(point);
        }
    }

    pub fn nearest_within(&self, point: &XYPoint32, radius: f32) -> Option<usize> {
        let (distance, idx) = self
            .point_index
            .nearest_one(&[point.x, point.y], &manhattan);
        if distance < radius {
            Some(idx)
        } else {
            None
        }
    }

    pub fn neighbor_indexes(&self, point: &XYPoint32, radius: f32) -> Vec<usize> {
        self.point_index
            .within_unsorted(&[point.x, point.y], radius, &manhattan)
            .iter()
            .map(|neighbor| neighbor.item)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ClusterClassification {
    Undefined,
    Noise,
    Border(usize),
    Core(usize),
}
