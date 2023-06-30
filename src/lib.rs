use std::collections::HashMap;

use pyo3::prelude::{pyfunction, pymodule, PyAny, PyErr, PyModule, PyObject, PyResult, Python};
use pyo3::types::{PyFloat, PyInt};
use pyo3::wrap_pyfunction;

use arrow::array::{make_array, Array, ArrayData, Float64Array, ListBuilder, UInt32Builder};
use arrow::error::ArrowError;
use arrow::pyarrow::{FromPyArrow, PyArrowException, ToPyArrow};

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
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
    py: Python,
) -> PyResult<PyObject> {
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

    // Turn xs and ys into Vec<XYPoint> for easier processing.
    let points = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| XYPoint {
            x: x.unwrap_or(0.0),
            y: y.unwrap_or(0.0),
        })
        .collect::<Vec<_>>();

    let quantized = quantize(points, eps);
    let map = hist2d(quantized);
    let mut builder = ListBuilder::new(UInt32Builder::new());
    for v in map.values() {
        if v.len() >= min_cluster_size {
            for i in v {
                builder.values().append_value(*i as u32);
            }
            builder.append(true);
        }
    }
    let la = builder.finish();
    la.to_data().to_pyarrow(py)
}

fn hist2d(points: Vec<XYPoint<i64>>) -> HashMap<XYPoint<i64>, Vec<usize>> {
    let mut map = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        map.entry(*p).or_insert_with(Vec::new).push(i);
    }
    map
}

/// Quantize points to a grid.
fn quantize(points: Vec<XYPoint<f64>>, quantum: f64) -> Vec<XYPoint<i64>> {
    points
        .iter()
        .map(|p| XYPoint {
            x: (p.x / quantum).round() as i64,
            y: (p.y / quantum).round() as i64,
        })
        .collect()
}

/// A point in 2D space.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct XYPoint<T> {
    x: T,
    y: T,
}

#[test]
fn test_quantize() {
    let points = vec![
        XYPoint { x: 0.0, y: 0.0 },
        XYPoint { x: 0.4, y: 0.4 },
        XYPoint { x: 1.0, y: 1.0 },
        XYPoint { x: 1.6, y: 1.6 },
        XYPoint { x: 2.0, y: 2.0 },
    ];
    let quantized = quantize(points, 1.0);
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
}

/// A Python module implemented in Rust.
#[pymodule]
fn thor_cluster(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_clusters_py, m)?)?;
    Ok(())
}
