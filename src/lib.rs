use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::{
    pyclass, pyfunction, pymodule, Py, PyAny, PyErr, PyModule, PyObject, PyResult, Python,
};
use pyo3::types::{PyFloat, PyInt, PyTuple};
use pyo3::wrap_pyfunction;

use arrow::array::{
    make_array, Array, ArrayData, Float64Array, Float64Builder, Int32Builder, StringArray,
    DictionaryArray, StringDictionaryBuilder,
    StringBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Schema, Int32Type};
use arrow::error::ArrowError;
use arrow::pyarrow::{FromPyArrow, PyArrowException, ToPyArrow};
use arrow::record_batch::RecordBatch;

use uuid;

mod dbscan;
pub mod gridsearch;
mod hotspot2d;
pub mod points;
use dbscan::fixed16_kdtree;
use dbscan::float32_kdtree;
use dbscan::rstar;

pub use points::{XYPoint, XYTPoint};

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

#[derive(Clone, PartialEq, Eq)]
#[pyclass]
pub enum ClusterAlgorithm {
    DBSCAN = 1,
    Hotspot2D = 2,
    DbscanRStar = 3,
    DbscanFixed16 = 4,
}

/// Clusters X-Y points, searching across a grid of possibly vx and vy values.
///
/// Arguments:
///     ids: A list of observation IDs as a StringArray.
///     xs: A list of x coordinates as a Float64Array.
///     ys: A list of y coordinates as a Float64Array.
///     dts: A list of time deltas as a Float64Array. These are the time deltas between
///          the point and the minimum time in the dataset, in MJD.
///     vxs: A list of possible x velocities as a Float64Array.
///     vys: A list of possible y velocities as a Float64Array.
///     eps: The maximum distance between two points for them to be considered in the same
///          neighborhood.
///     min_cluster_size: The minimum number of points in a cluster.
///     n_threads: The number of threads to use for clustering.
///     alg: The clustering algorithm to use.
///
/// Returns:
///     A pair of RecordBatches.
///     The first summarizes all of the clusters. It has the following schema:
///         cluster_id: string
///         vx: float64
///         vy: float64
///         arc_length: float64
///     The second contains the cluster assignments for each point. It
///     has the following schema:
///         cluster_id: string
///         obs_id: string
#[pyfunction]
#[pyo3(name = "grid_search")]
fn grid_search_py(
    ids: &PyAny,
    xs: &PyAny,
    ys: &PyAny,
    dts: &PyAny,
    vxs: &PyAny,
    vys: &PyAny,
    eps: &PyFloat,
    min_cluster_size: &PyInt,
    n_threads: &PyInt,
    alg: Py<ClusterAlgorithm>,
    py: Python,
) -> PyResult<PyObject> {
    // Handle the Python-to-rust conversion up front
    let ids = make_array(ArrayData::from_pyarrow(ids)?);
    let ids = ids
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ArrowError::ParseError("Expects a string array".to_string()))
        .map_err(to_py_err)?;

    let xs = make_array(ArrayData::from_pyarrow(xs)?);

    let xs = xs
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    let ys = make_array(ArrayData::from_pyarrow(ys)?);

    let ys = ys
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    let dts = make_array(ArrayData::from_pyarrow(dts)?);
    let dts = dts
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    if xs.len() != ys.len() || xs.len() != dts.len() {
        return Err(PyArrowException::new_err(
            "x y, and dts arrays must be the same length",
        ));
    }

    let vxs = make_array(ArrayData::from_pyarrow(vxs)?);
    let vxs = vxs
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    let vys = make_array(ArrayData::from_pyarrow(vys)?);
    let vys = vys
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    let eps = eps.extract::<f64>()?;
    let min_cluster_size = min_cluster_size.extract::<u8>()? as usize;
    let alg = alg.extract::<ClusterAlgorithm>(py)?;
    let n_threads = n_threads.extract::<usize>()?;

    // Turn xs ys, and dts into Vec<XYTPoint> for easier processing.
    let points = xs
        .iter()
        .zip(ys.iter())
        .zip(dts.iter())
        .map(|((x, y), dt)| XYTPoint::new(x.unwrap_or(0.0), y.unwrap_or(0.0), dt.unwrap_or(0.0)))
        .collect::<Vec<_>>();

    // Turn vxs and vys into Vec<f64> for easier processing.
    let vxs = vxs.iter().map(|x| x.unwrap_or(0.0)).collect::<Vec<_>>();
    let vys = vys.iter().map(|x| x.unwrap_or(0.0)).collect::<Vec<_>>();

    let results =
        gridsearch::cluster_grid_search(&points, vxs, vys, alg, eps, min_cluster_size, n_threads);

    // Result shape is a pair of values.
    //
    // The first value is a table of cluster ID, vx, vy, and arc length (difference between min and max dt).
    //
    // The second value is a table of cluster IDs and observation IDs.
    let cluster_table_schema = Schema::new(vec![
        Field::new("cluster_id", DataType::UInt32, false),
        Field::new("vx", DataType::Float64, false),
        Field::new("vy", DataType::Float64, false),
        Field::new("arc_length", DataType::Float64, false),
    ]);

    let cluster_members_table_schema = Schema::new(vec![
        Field::new("cluster_id", DataType::UInt32, false),
        Field::new_dictionary("obs_id", DataType::Int32, DataType::Utf8, false),
    ]);

    // Assemble the arrays.
    let mut cluster_id_builder = UInt32Builder::new();
    let mut vx_builder = Float64Builder::new();
    let mut vy_builder = Float64Builder::new();
    let mut arc_length_builder = Float64Builder::new();

    let mut cluster_id_members_builder = UInt32Builder::new();
    let mut obs_id_members_builder = StringDictionaryBuilder::<Int32Type>::new();

    let mut cluster_id: u32 = 0;
    for result in results.into_iter() {
        let mut label_id_map: HashMap<i32, u32> = HashMap::new();
        let mut cluster_arc_starts: HashMap<u32, f64> = HashMap::new();
        let mut cluster_arc_ends: HashMap<u32, f64> = HashMap::new();
        let mut cluster_ids = Vec::new();
        for (i, label) in result.cluster_labels.iter().enumerate() {
            if *label < 0 {
                continue;
            }
            let cluster_id = match label_id_map.get(label) {
                Some(val) => *val,
                None => {
                    let val = cluster_id + 1;
		    cluster_id += 1;
                    label_id_map.insert(*label, val);
                    cluster_ids.push(val);
                    vx_builder.append_value(result.vx);
                    vy_builder.append_value(result.vy);
                    val
                }
            };
            cluster_id_members_builder.append_value(cluster_id);
            obs_id_members_builder.append_value(ids.value(i));
            // Keep track of max/min dt for each cluster.
            let dt = dts.value(i);
            match cluster_arc_starts.get(&cluster_id) {
                Some(val) => {
                    if dt < *val {
                        cluster_arc_starts.insert(cluster_id, dt);
                    }
                }
                None => {
                    cluster_arc_starts.insert(cluster_id, dt);
                }
            };
            match cluster_arc_ends.get(&cluster_id) {
                Some(val) => {
                    if dt > *val {
                        cluster_arc_ends.insert(cluster_id, dt);
                    }
                }
                None => {
                    cluster_arc_ends.insert(cluster_id, dt);
                }
            }
        }
        // Now that we've processed all the points, we can add the arc lengths.
        for cluster_id in cluster_ids.iter() {
            cluster_id_builder.append_value(*cluster_id);
            let arc_length = cluster_arc_ends.get(cluster_id).unwrap()
                - cluster_arc_starts.get(cluster_id).unwrap();
            arc_length_builder.append_value(arc_length);
        }
    }

    // Build the tables (as RecordBatches)
    let cluster_table = RecordBatch::try_new(
        Arc::new(cluster_table_schema),
        vec![
            Arc::new(cluster_id_builder.finish()),
            Arc::new(vx_builder.finish()),
            Arc::new(vy_builder.finish()),
            Arc::new(arc_length_builder.finish()),
        ],
    )
    .map_err(to_py_err)?;

    let cluster_members_table = RecordBatch::try_new(
        Arc::new(cluster_members_table_schema),
        vec![
            Arc::new(cluster_id_members_builder.finish()),
            Arc::new(obs_id_members_builder.finish()),
        ],
    )
    .map_err(to_py_err)?;

    // Convert to Python objects for output
    let cluster_table = cluster_table.to_pyarrow(py)?;
    let cluster_members_table = cluster_members_table.to_pyarrow(py)?;

    // Combine into a tuple.
    Ok(PyTuple::new(py, vec![cluster_table, cluster_members_table]).into())
}

/// Find clusters of related x-y points.
///
/// # Arguments
///
/// * `xs` - A arrow float64 array of x values.
/// * `ys` - A arrow float64 array of y values.
/// * `eps` - The maximum distance between two points for them to be considered as in the same cluster.
/// * `min_cluster_size` - The minimum number of points in a cluster.
///
/// # Returns
///
/// A list of lists of indices into the input arrays, as an arrow list of uint32 arrays.
#[pyfunction]
#[pyo3(name = "find_clusters")]
fn find_clusters_py(
    xs: &PyAny,
    ys: &PyAny,
    eps: &PyFloat,
    min_cluster_size: &PyInt,
    alg: Py<ClusterAlgorithm>,
    py: Python,
) -> PyResult<PyObject> {
    // Handle the Python-to-rust conversion up front
    let xs = make_array(ArrayData::from_pyarrow(xs)?);

    let xs = xs
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    let ys = make_array(ArrayData::from_pyarrow(ys)?);

    let ys = ys
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| ArrowError::ParseError("Expects a float64 array".to_string()))
        .map_err(to_py_err)?;

    if xs.len() != ys.len() {
        return Err(PyArrowException::new_err(
            "x and y arrays must be the same length",
        ));
    }

    let eps = eps.extract::<f64>()?;
    let min_cluster_size = min_cluster_size.extract::<u8>()? as usize;
    let alg = alg.extract::<ClusterAlgorithm>(py)?;

    // Turn xs and ys into Vec<XYPoint> for easier processing.
    let points = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| XYPoint {
            x: x.unwrap_or(0.0),
            y: y.unwrap_or(0.0),
        })
        .collect::<Vec<_>>();

    let cluster_labels = find_clusters(&points, eps, min_cluster_size, &alg);

    // Convert the clusters into an arrow list of uint32 arrays.
    let mut builder = Int32Builder::new();
    builder.append_slice(&cluster_labels[..]);
    let la = builder.finish();
    la.to_data().to_pyarrow(py)
}

pub fn find_clusters(
    points: &Vec<XYPoint<f64>>,
    eps: f64,
    min_cluster_size: usize,
    alg: &ClusterAlgorithm,
) -> Vec<i32> {
    match alg {
        ClusterAlgorithm::Hotspot2D => {
            hotspot2d::find_clusters_hotspot2d(points, eps, min_cluster_size)
        }
        ClusterAlgorithm::DBSCAN => {
            dbscan::find_clusters::<float32_kdtree::PointTree>(points, eps, min_cluster_size)
        }
        ClusterAlgorithm::DbscanRStar => {
            dbscan::find_clusters::<rstar::Tree>(points, eps, min_cluster_size)
        }
        ClusterAlgorithm::DbscanFixed16 => {
            dbscan::find_clusters::<fixed16_kdtree::FixedPointTree>(points, eps, min_cluster_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_clusters_near_miss() {
        let points = vec![
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 1.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
        ];
        let clusters = find_clusters(&points, 1.0, 4, &ClusterAlgorithm::Hotspot2D);
        let expect = vec![-1, -1, -1, -1, 0, 0, 0, 0];
        assert_eq!(clusters, expect);
    }

    #[test]
    fn test_find_clusters_two_hits() {
        let points = vec![
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
        ];
        let clusters = find_clusters(&points, 1.0, 4, &ClusterAlgorithm::DBSCAN);
        let allowed = vec![vec![1, 1, 1, 1, 2, 2, 2, 2], vec![2, 2, 2, 2, 1, 1, 1, 1]];
        assert!(allowed.contains(&clusters));
    }

    #[test]
    fn test_find_clusters_bigger_than_min() {
        let points = vec![
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(1.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
            XYPoint::new(2.0, 0.0),
        ];
        let clusters = find_clusters(&points, 1.0, 2, &ClusterAlgorithm::DBSCAN);
        let allowed = vec![vec![1, 1, 1, 1, 2, 2, 2, 2], vec![2, 2, 2, 2, 1, 1, 1, 1]];
        assert!(allowed.contains(&clusters));
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn thor_cluster(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_clusters_py, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_py, m)?)?;
    m.add_class::<ClusterAlgorithm>()?;
    Ok(())
}
