use pyo3::prelude::{
    pyclass, pyfunction, pymodule, Py, PyAny, PyErr, PyModule, PyObject, PyResult, Python,
};
use pyo3::types::{PyFloat, PyInt};
use pyo3::wrap_pyfunction;

use arrow::array::{make_array, Array, ArrayData, Float64Array, Int32Builder};
use arrow::error::ArrowError;
use arrow::pyarrow::{FromPyArrow, PyArrowException, ToPyArrow};

mod dbscan;
mod hotspot2d;
pub mod points;

pub use points::XYPoint;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

#[derive(Clone, PartialEq, Eq)]
#[pyclass]
pub enum ClusterAlgorithm {
    DBSCAN = 1,
    Hotspot2D = 2,
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

    let cluster_labels = find_clusters(points, eps, min_cluster_size, alg);

    // Convert the clusters into an arrow list of uint32 arrays.
    let mut builder = Int32Builder::new();
    builder.append_slice(&cluster_labels[..]);
    let la = builder.finish();
    la.to_data().to_pyarrow(py)
}

pub fn find_clusters(
    points: Vec<XYPoint<f64>>,
    eps: f64,
    min_cluster_size: usize,
    alg: ClusterAlgorithm,
) -> Vec<i32> {
    match alg {
        ClusterAlgorithm::DBSCAN => dbscan::find_clusters_dbscan(points, eps, min_cluster_size),
        ClusterAlgorithm::Hotspot2D => {
            hotspot2d::find_clusters_hotspot2d(points, eps, min_cluster_size)
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
        let clusters = find_clusters(points, 1.0, 4, ClusterAlgorithm::Hotspot2D);
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
        let clusters = find_clusters(points, 1.0, 4, ClusterAlgorithm::DBSCAN);
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
        let clusters = find_clusters(points, 1.0, 2, ClusterAlgorithm::DBSCAN);
        let allowed = vec![vec![1, 1, 1, 1, 2, 2, 2, 2], vec![2, 2, 2, 2, 1, 1, 1, 1]];
        assert!(allowed.contains(&clusters));
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn thor_cluster(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_clusters_py, m)?)?;
    m.add_class::<ClusterAlgorithm>()?;
    Ok(())
}
