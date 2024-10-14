use anyhow::anyhow;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyComplex};
use spenso::{
    complex::RealOrComplex, data::{GetTensorData, SetTensorData}, network::TensorNetwork, parametric::{ConcreteOrParam, MixedTensor, SerializableAtom}, structure::{AtomStructure, HasStructure, TensorStructure}
};
use symbolica::{api::python::PythonExpression, atom::Atom};

#[pyclass(name = "TensorNetwork", module = "symbolica", subclass)]
#[derive(Clone)]
pub struct SpensoNet {
    pub network: TensorNetwork<MixedTensor<f64, AtomStructure>, Atom>,
}

#[pyclass(name = "Tensor", module = "symbolica", subclass)]
#[derive(Clone)]
pub struct Spensor {
    pub tensor: MixedTensor<f64, AtomStructure>,
}


#[pymethods]
impl Spensor {
    fn scalar(&self) -> PyResult<PythonExpression> {
        self.tensor.clone().scalar().map(|r|PythonExpression { expr:  r.into()}).ok_or_else(||PyRuntimeError::new_err("No scalar found"))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.tensor)
    }

    fn __str__(&self) -> String {
        format!("{}", self.tensor)
    }

    fn __len__(&self)->usize{
        self.tensor.size().unwrap()
    }

    fn __getitem__(&self, item: &PyAny)-> anyhow::Result<Py<PyAny>>{
        let out = if let Ok(flat_index) = item.extract::<usize>(){
             self.tensor.get_owned_linear(flat_index.into()).ok_or(anyhow!("index out of bounds"))?
        } else if let Ok(expanded_idxs)= item.extract::<Vec<usize>>() {
            self.tensor.get_owned(&expanded_idxs)?
        }else {
            return Err(anyhow!("Index must be an integer"));
        };

        match out {
           ConcreteOrParam::Concrete(c)=>{
              match c{

               RealOrComplex::Complex(c)=> {
                   Ok(Python::with_gil(|py|PyComplex::from_doubles(py,c.re,c.im).to_object(py)))
               }
               RealOrComplex::Real(r)=>{
                   Ok(Python::with_gil(|py| r.into_py(py)))
               }
              }
           }
           ConcreteOrParam::Param(p)=>{
               Ok(Python::with_gil(|py| PythonExpression{expr:p}.into_py(py)))
           }
       }
    }

    fn __setitem__(&mut self, item: &PyAny,value:&PyAny)-> anyhow::Result<()>{

        let value = if let Ok(v)= value.extract::<PythonExpression>(){
            ConcreteOrParam::Param(v.expr)} else if let Ok(v)= value.extract::<f64>(){
                ConcreteOrParam::Concrete(RealOrComplex::Real(v))}
            else{
                return Err(anyhow!("Value must be a PythonExpression or a float"));
            };

        if let Ok(flat_index) = item.extract::<usize>(){
            self.tensor.set_flat(flat_index.into(),value)
        } else if let Ok(expanded_idxs)= item.extract::<Vec<usize>>() {
            self.tensor.set(&expanded_idxs,value)
        }else {
            Err(anyhow!("Index must be an integer"))
        }
    }
}



pub type ParsingNet = TensorNetwork<MixedTensor<f64, AtomStructure>, SerializableAtom>;

#[pymethods]
impl SpensoNet {
    #[new]
    pub fn from_expression(a: PythonExpression) -> anyhow::Result<SpensoNet>{
        Ok(SpensoNet {
            network: ParsingNet::try_from(a.expr.as_view())?.map_scalar(|r| r.0),
        })
    }

    fn contract(&mut self) -> PyResult<()> {
        self.network.contract();
        Ok(())
    }

    fn result(&self) -> PyResult<Spensor> {
        Ok(Spensor {
            tensor: self
                .network
                .result_tensor_smart()
                .map_err(|s| PyRuntimeError::new_err(s.to_string()))?,
        })
    }
}
