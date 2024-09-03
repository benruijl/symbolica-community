use pyo3::prelude::*;

pub mod physics;

#[pymodule]
fn symbolica_community(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    symbolica::api::python::create_symbolica_module(m)?;

    // register new components
    physics::initialize(m)?;

    Ok(())
}
