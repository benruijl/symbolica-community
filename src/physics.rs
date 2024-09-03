pub mod trace;

use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyResult};
use symbolica::api::python::PythonExpression;

/// Compute the Diract trace.
#[pyfunction(name = "trace")]
fn python_trace(a: PythonExpression) -> PythonExpression {
    trace::trace(a.expr.as_view()).into()
}

pub(crate) fn initialize(m: &PyModule) -> PyResult<()> {
    m.getattr("Expression")?
        .setattr("trace", wrap_pyfunction!(python_trace, m)?)
}
