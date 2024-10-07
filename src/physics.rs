pub mod trace;
#[cfg(feature = "vakint")]
pub mod vakint;
#[cfg(feature = "vakint")]
use vakint::{
    NumericalEvaluationResultWrapper, VakintEvaluationMethodWrapper, VakintExpressionWrapper,
    VakintWrapper,
};

use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyResult};
use symbolica::api::python::PythonExpression;

/// Compute the Diract trace.
#[pyfunction(name = "trace")]
fn python_trace(a: PythonExpression) -> PythonExpression {
    trace::trace(a.expr.as_view()).into()
}

pub(crate) fn initialize(m: &PyModule) -> PyResult<()> {
    m.getattr("Expression")?
        .setattr("trace", wrap_pyfunction!(python_trace, m)?)?;

    #[cfg(feature = "vakint")]
    {
        m.add_class::<VakintWrapper>()?;
        m.add_class::<NumericalEvaluationResultWrapper>()?;
        m.add_class::<VakintExpressionWrapper>()?;
        m.add_class::<VakintEvaluationMethodWrapper>()?;
    }
    Ok(())
}
