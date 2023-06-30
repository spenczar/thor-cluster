use crate::points::XYPoint;
use std::collections::HashMap;

pub fn find_clusters_hotspot2d(
    points: Vec<XYPoint<f64>>,
    eps: f64,
    min_cluster_size: usize,
) -> Vec<i32> {
    // Run 4 times with different quantization to catch near misses.
    let quantized1 = quantize(&points, eps);
    let map1 = hist2d(&quantized1);
    let labels1 = label_cluster_map(&quantized1, map1, min_cluster_size);

    let points2 = &points
        .iter()
        .map(|p| XYPoint {
            x: p.x + eps / 2.0,
            y: p.y,
        })
        .collect::<Vec<_>>();
    let quantized2 = quantize(points2, eps);
    let map2 = hist2d(&quantized2);
    let labels2 = label_cluster_map(&quantized2, map2, min_cluster_size);

    let points3 = &points
        .iter()
        .map(|p| XYPoint {
            x: p.x,
            y: p.y + eps / 2.0,
        })
        .collect::<Vec<_>>();
    let quantized3 = quantize(points3, eps);
    let map3 = hist2d(&quantized3);
    let labels3 = label_cluster_map(&quantized3, map3, min_cluster_size);

    let points4 = &points
        .iter()
        .map(|p| XYPoint {
            x: p.x + eps / 2.0,
            y: p.y + eps / 2.0,
        })
        .collect::<Vec<_>>();
    let quantized4 = quantize(points4, eps);
    let map4 = hist2d(&quantized4);
    let labels4 = label_cluster_map(&quantized4, map4, min_cluster_size);

    merge_cluster_labels(&labels1, &labels2, &labels3, &labels4)
}

pub fn merge_cluster_labels(
    l1: &Vec<i32>,
    l2: &Vec<i32>,
    l3: &Vec<i32>,
    l4: &Vec<i32>,
) -> Vec<i32> {
    let mut labels = vec![0; l1.len()];
    for (i, l) in l1.iter().enumerate() {
        if *l != -1 {
            labels[i] = *l;
        } else if l2[i] != -1 {
            labels[i] = l2[i];
        } else if l3[i] != -1 {
            labels[i] = l3[i];
        } else if l4[i] != -1 {
            labels[i] = l4[i];
        } else {
            labels[i] = -1;
        }
    }
    labels
}

/// Mark points as belonging to a cluster. A value of -1 means the
/// point is not in a cluster.
///
pub fn label_cluster_map(
    points: &Vec<XYPoint<i64>>,
    cluster_map: HashMap<XYPoint<i64>, Vec<usize>>,
    min_size: usize,
) -> Vec<i32> {
    let mut labels = vec![0; points.len()];
    let mut label_map = HashMap::new();
    let mut label = 0;
    cluster_map.iter().for_each(|(p, v)| {
        if v.len() >= min_size {
            label_map.insert(p, label);
            label += 1;
        }
    });

    for (i, p) in points.iter().enumerate() {
        let v = label_map.get(p);
        if let Some(group) = v {
            labels[i] = *group;
        } else {
            labels[i] = -1;
        }
    }
    labels
}

pub fn hist2d(points: &Vec<XYPoint<i64>>) -> HashMap<XYPoint<i64>, Vec<usize>> {
    let mut map = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        map.entry(*p).or_insert_with(Vec::new).push(i);
    }
    map
}

/// Quantize points to a grid.
pub fn quantize(points: &Vec<XYPoint<f64>>, quantum: f64) -> Vec<XYPoint<i64>> {
    points
        .iter()
        .map(|p| XYPoint {
            x: (p.x / quantum).round() as i64,
            y: (p.y / quantum).round() as i64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize() {
        let points = vec![
            XYPoint { x: 0.0, y: 0.0 },
            XYPoint { x: 0.4, y: 0.4 },
            XYPoint { x: 1.0, y: 1.0 },
            XYPoint { x: 1.6, y: 1.6 },
            XYPoint { x: 2.0, y: 2.0 },
        ];
        let quantized = quantize(&points, 1.0);
        assert_eq!(
            quantized,
            vec![
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 1, y: 1 },
                XYPoint { x: 2, y: 2 },
                XYPoint { x: 2, y: 2 },
            ]
        );

        let quantized = quantize(&points, 0.5);
        assert_eq!(
            quantized,
            vec![
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 1, y: 1 },
                XYPoint { x: 2, y: 2 },
                XYPoint { x: 3, y: 3 },
                XYPoint { x: 4, y: 4 },
            ]
        );
    }

    #[test]
    fn test_quantize_negative() {
        let points = vec![
            XYPoint { x: -1.4, y: -1.4 },
            XYPoint { x: -0.1, y: -0.1 },
            XYPoint { x: 0.0, y: 0.0 },
            XYPoint { x: 0.1, y: 0.1 },
            XYPoint { x: 0.4, y: 0.4 },
            XYPoint { x: 1.0, y: 1.0 },
        ];
        let quantized = quantize(&points, 1.0);
        assert_eq!(
            quantized,
            vec![
                XYPoint { x: -1, y: -1 },
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 0, y: 0 },
                XYPoint { x: 1, y: 1 },
            ]
        );
    }

    #[test]
    fn test_quantize_empty() {
        let points = vec![];
        let quantized = quantize(&points, 1.0);
        assert_eq!(quantized, vec![]);
    }

    #[test]
    fn test_hist2d() {
        let points = vec![
            XYPoint { x: 0, y: 0 },
            XYPoint { x: 0, y: 0 },
            XYPoint { x: 1, y: 1 },
            XYPoint { x: 2, y: 2 },
            XYPoint { x: 2, y: 2 },
        ];
        let map = hist2d(&points);
        assert_eq!(map.len(), 3);
        assert_eq!(map[&XYPoint { x: 0, y: 0 }], vec![0, 1]);
        assert_eq!(map[&XYPoint { x: 1, y: 1 }], vec![2]);
        assert_eq!(map[&XYPoint { x: 2, y: 2 }], vec![3, 4]);
    }

    #[test]
    fn test_hist2d_empty() {
        let points = vec![];
        let map = hist2d(&points);
        assert_eq!(map.len(), 0);
    }
}
