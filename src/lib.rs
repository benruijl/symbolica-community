use pyo3::prelude::*;

pub mod physics;

#[pymodule]
fn symbolica_community(m: &Bound<'_, PyModule>) -> PyResult<()> {
    symbolica::api::python::create_symbolica_module(m)?;

    // register new components
    physics::initialize(m)?;

    Ok(())
}
