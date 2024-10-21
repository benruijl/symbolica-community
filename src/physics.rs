pub mod trace;
#[cfg(feature = "vakint")]
pub mod vakint;
#[cfg(feature = "vakint")]
use vakint::{
    NumericalEvaluationResultWrapper, VakintEvaluationMethodWrapper, VakintExpressionWrapper,
    VakintWrapper,
};
#[cfg(feature = "spenso")]
pub mod tensors;

#[cfg(feature = "spenso")]
use tensors::SpensoNet;

use pyo3::{
    pyfunction,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};
use symbolica::api::python::PythonExpression;

/// Compute the Diract trace.
#[pyfunction(name = "trace")]
fn python_trace(a: PythonExpression) -> PythonExpression {
    trace::trace(a.expr.as_view()).into()
}

#[cfg(feature = "spenso")]
#[pyfunction(name = "to_net")]
pub fn python_to_tensor_network(a: PythonExpression) -> anyhow::Result<SpensoNet> {
    SpensoNet::from_expression(a)
}

pub(crate) fn initialize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("Expression")?
        .setattr("trace", wrap_pyfunction!(python_trace, m)?)?;

    #[cfg(feature = "spenso")]
    {
        m.getattr("Expression")?
            .setattr("to_net", wrap_pyfunction!(python_to_tensor_network, m)?)?;
        m.add_class::<SpensoNet>()?;
        m.add_class::<tensors::Spensor>()?;
    }

    #[cfg(feature = "vakint")]
    {
        m.add_class::<VakintWrapper>()?;
        m.add_class::<NumericalEvaluationResultWrapper>()?;
        m.add_class::<VakintExpressionWrapper>()?;
        m.add_class::<VakintEvaluationMethodWrapper>()?;
    }
    Ok(())
}
