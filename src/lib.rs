use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::{
    pyfunction, pymodule, PyAny, PyErr, PyModule, PyObject, PyResult, Python, ToPyObject,
};
use pyo3::types::{PyFloat, PyInt};
use pyo3::wrap_pyfunction;

use arrow::array::{
    make_array, Array, ArrayData, ArrayRef, Float64Array, Int64Array, ListArray, ListBuilder,
    UInt32Builder,
};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow::pyarrow::{FromPyArrow, PyArrowException, ToPyArrow};
use arrow_row::{Row, RowConverter, Rows, SortField};

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

type Result<T> = std::result::Result<T, PyErr>;

#[pyfunction]
fn find_clusters(
    xs: &PyAny,
    ys: &PyAny,
    eps: &PyFloat,
    min_sample: &PyInt,
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

    let eps = eps.extract::<f64>()?;
    let quantized_x = quantize(&xs, eps);
    let quantized_y = quantize(&ys, eps);

    // Bundle into rows.
    let mut converter = RowConverter::new(vec![
        SortField::new(DataType::Int64),
        SortField::new(DataType::Int64),
    ])
    .map_err(to_py_err)?;

    let xref = Arc::new(quantized_x.clone()) as ArrayRef;
    let yref = Arc::new(quantized_y.clone()) as ArrayRef;
    let cols = vec![xref, yref];
    let rows = converter.convert_columns(&cols).map_err(to_py_err)?;

    if true {
        let min_sample = min_sample.extract::<u8>()? as usize;
        let mut map = hist2d(&rows);
        let mut builder = ListBuilder::new(UInt32Builder::new());
        for v in map.values() {
            if v.len() >= min_sample {
                for i in v {
                    builder.values().append_value(*i as u32);
                }
                builder.append(true);
            }
        }
        let la = builder.finish();
        return Ok(la.to_data().to_pyarrow(py)?);
    }

    // Sort the rows.
    let mut sort: Vec<_> = rows.iter().enumerate().collect();
    sort.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));

    // Find runs in the sorted rows.
    let order = sort.iter().map(|(i, _)| *i).collect::<Vec<_>>();
    let runs = find_runs(order, &rows, min_sample.extract::<u8>()?);
    let la = vec_to_arrow(runs);
    Ok(la.to_data().to_pyarrow(py)?)
}

fn vec_to_arrow(vals: Vec<Vec<usize>>) -> ListArray {
    let mut builder = ListBuilder::new(UInt32Builder::new());
    for v in vals {
        for i in v {
            builder.values().append_value(i as u32);
        }
        builder.append(true);
    }
    builder.finish()
}

fn hist2d(rows: &Rows) -> HashMap<Row, Vec<usize>> {
    let mut map = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        let entry = map.entry(row.clone()).or_insert(vec![]);
        entry.push(i);
    }
    map
}

fn find_runs(order: Vec<usize>, rows: &Rows, min_samples: u8) -> Vec<Vec<usize>> {
    let mut runs = vec![];
    let mut run = vec![];
    let mut last_row = rows.row(order[0]);
    for &i in order.iter() {
        let r = rows.row(i);
        if r == last_row {
            run.push(i);
        } else {
            if run.len() >= min_samples as usize {
                runs.push(run);
            }
            run = vec![i];
            last_row = r;
        }
    }
    if run.len() >= min_samples as usize {
        runs.push(run);
    }
    runs
}

fn quantize(vals: &Float64Array, quantum: f64) -> Int64Array {
    let vec = vals
        .iter()
        .map(|x| (x.unwrap_or(0.0) / quantum).round() as i64)
        .collect::<Vec<_>>();
    return Int64Array::from(vec);
}

/// A Python module implemented in Rust.
#[pymodule]
fn thor_cluster(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_clusters, m)?)?;
    Ok(())
}
