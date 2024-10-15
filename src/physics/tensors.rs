use std::collections::HashMap;

use anyhow::anyhow;
use pyo3::{
    exceptions::{self, PyIndexError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::PyComplex,
};

use spenso::{
    complex::{RealOrComplex, RealOrComplexTensor},
    data::{DataTensor, GetTensorData, SetTensorData},
    network::TensorNetwork,
    parametric::{
        CompiledEvalTensor, ConcreteOrParam, LinearizedEvalTensor, MixedTensor, ParamOrConcrete,
    },
    structure::{AtomStructure, HasStructure, TensorStructure},
    symbolica_utils::SerializableAtom,
};
use symbolica::{
    api::python::PythonExpression,
    atom::Atom,
    domains::float::Complex,
    evaluate::{CompileOptions, FunctionMap, InlineASM, OptimizationSettings},
    poly::Variable,
};

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

impl From<DataTensor<f64, AtomStructure>> for Spensor {
    fn from(value: DataTensor<f64, AtomStructure>) -> Self {
        Self {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Real(value)),
        }
    }
}

impl From<DataTensor<Complex<f64>, AtomStructure>> for Spensor {
    fn from(value: DataTensor<Complex<f64>, AtomStructure>) -> Self {
        Self {
            tensor: MixedTensor::Concrete(RealOrComplexTensor::Complex(
                value.map_data(|c| c.into()),
            )),
        }
    }
}

/// An optimized evaluator for expressions.
#[pyclass(name = "TensorEvaluator", module = "symbolica")]
#[derive(Clone)]
pub struct SpensoExpressionEvaluator {
    pub eval: LinearizedEvalTensor<f64, AtomStructure>,
}

/// A compiled and optimized evaluator for expressions.
#[pyclass(name = "CompiledEvaluator", module = "symbolica")]
#[derive(Clone)]
pub struct SpensoCompiledExpressionEvaluator {
    pub eval: CompiledEvalTensor<AtomStructure>,
}

#[pymethods]
impl SpensoCompiledExpressionEvaluator {
    // /// Load a compiled library, previously generated with `compile`.
    // #[classmethod]
    // fn load(
    //     _cls: &PyType,
    //     filename: &str,
    //     function_name: &str,
    //     input_len: usize,
    //     output_len: usize,
    // ) -> PyResult<Self> {
    //     Ok(Self {
    //         eval: CompiledEvalTensor::load(filename, function_name)
    //             .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
    //         input_len,
    //         output_len,
    //     })
    // }

    // /// Evaluate the expression for multiple inputs that are flattened and return the flattened result.
    // /// This method has less overhead than `evaluate`.
    // fn evaluate_flat(&mut self, inputs: Vec<f64>) -> Vec<f64> {
    //     let n_inputs = inputs.len() / self.input_len;
    //     let mut res = vec![0.; self.output_len * n_inputs];
    //     for (r, s) in res
    //         .chunks_mut(self.output_len)
    //         .zip(inputs.chunks(self.input_len))
    //     {
    //         self.eval.evaluate(s, r);
    //     }

    //     res
    // }

    // /// Evaluate the expression for multiple inputs that are flattened and return the flattened result.
    // /// This method has less overhead than `evaluate_complex`.
    // fn evaluate_complex_flat<'py>(
    //     &mut self,
    //     py: Python<'py>,
    //     inputs: Vec<Complex<f64>>,
    // ) -> Vec<&'py PyComplex> {
    //     let n_inputs = inputs.len() / self.input_len;
    //     let mut res = vec![PyComplex::from_doubles(py, 0., 0.); self.output_len * n_inputs];
    //     let mut tmp = vec![Complex::new_zero(); self.output_len];
    //     for (r, s) in res
    //         .chunks_mut(self.output_len)
    //         .zip(inputs.chunks(self.input_len))
    //     {
    //         self.eval.evaluate(s, &mut tmp);
    //         for (rr, t) in r.iter_mut().zip(&tmp) {
    //             *rr = PyComplex::from_doubles(py, t.re, t.im);
    //         }
    //     }

    //     res
    // }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }
}

#[pymethods]
impl SpensoExpressionEvaluator {
    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate(&mut self, inputs: Vec<Vec<f64>>) -> Vec<Spensor> {
        inputs
            .iter()
            .map(|s| self.eval.evaluate(s).into())
            .collect()
    }

    /// Evaluate the expression for multiple inputs and return the results.
    fn evaluate_complex(&mut self, inputs: Vec<Vec<Complex<f64>>>) -> Vec<Spensor> {
        let mut eval = self.eval.clone().map_coeff(&|x| Complex::new(*x, 0.));

        inputs.iter().map(|s| eval.evaluate(s).into()).collect()
    }

    /// Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
    #[pyo3(signature =
        (function_name,
        filename,
        library_name,
        inline_asm = true,
        optimization_level = 3,
        compiler_path = None,
    ))]
    fn compile(
        &self,
        function_name: &str,
        filename: &str,
        library_name: &str,
        inline_asm: bool,
        optimization_level: u8,
        compiler_path: Option<&str>,
    ) -> PyResult<SpensoCompiledExpressionEvaluator> {
        let mut options = CompileOptions {
            optimization_level: optimization_level as usize,
            ..Default::default()
        };

        if let Some(compiler_path) = compiler_path {
            options.compiler = compiler_path.to_string();
        }

        Ok(SpensoCompiledExpressionEvaluator {
            eval: self
                .eval
                .export_cpp(
                    filename,
                    function_name,
                    true,
                    if inline_asm {
                        InlineASM::X64
                    } else {
                        InlineASM::None
                    },
                )
                .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                .compile(library_name, options)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                })?
                .load()
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                })?,
        })
    }
}

#[pymethods]
impl Spensor {
    #[pyo3(signature =
           (constants,
           functions,
           params,
           iterations = 100,
           n_cores = 4,
           verbose = false),
           )]
    pub fn evaluator(
        &self,
        constants: HashMap<PythonExpression, PythonExpression>,
        functions: HashMap<(Variable, String, Vec<Variable>), PythonExpression>,
        params: Vec<PythonExpression>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) -> PyResult<SpensoExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for (k, v) in &constants {
            if let Ok(r) = v.expr.clone().try_into() {
                fn_map.add_constant(k.expr.as_view(), r);
            } else {
                Err(exceptions::PyValueError::new_err(
                               "Constants must be rationals. If this is not possible, pass the value as a parameter",
                           ))?
            }
        }

        for ((symbol, rename, args), body) in &functions {
            let symbol = symbol
                .to_id()
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Bad function name {}",
                    symbol
                )))?;
            let args: Vec<_> = args
                .iter()
                .map(|x| {
                    x.to_id().ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {}",
                        symbol
                    )))
                })
                .collect::<Result<_, _>>()?;

            fn_map
                .add_function(symbol, rename.clone(), args, body.expr.as_view())
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Could not add function: {}", e))
                })?;
        }

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            n_cores,
            verbose,
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        // let eval = self
        //     .expr
        //     .evaluator(&fn_map, &params, settings)
        //     .map_err(|e| {
        //         exceptions::PyValueError::new_err(format!("Could not create evaluator: {}", e))
        //     })?;

        let mut evaltensor = match &self.tensor {
            ParamOrConcrete::Param(s) => s.eval_tree(&fn_map, &params).map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not create evaluator: {}", e))
            })?,
            ParamOrConcrete::Concrete(_) => return Err(PyRuntimeError::new_err("not atom")),
        };

        evaltensor.optimize_horner_scheme(
            settings.horner_iterations,
            settings.n_cores,
            settings.hot_start.clone(),
            settings.verbose,
        );

        evaltensor.common_subexpression_elimination();
        let linear = evaltensor.linearize(None);
        Ok(SpensoExpressionEvaluator {
            eval: linear.map_coeff(&|x| x.to_f64()),
        })
    }

    fn scalar(&self) -> PyResult<PythonExpression> {
        self.tensor
            .clone()
            .scalar()
            .map(|r| PythonExpression { expr: r.into() })
            .ok_or_else(|| PyRuntimeError::new_err("No scalar found"))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.tensor)
    }

    fn __str__(&self) -> String {
        format!("{}", self.tensor)
    }

    fn __len__(&self) -> usize {
        self.tensor.size().unwrap()
    }

    fn __getitem__(&self, item: &PyAny) -> PyResult<Py<PyAny>> {
        let out = if let Ok(flat_index) = item.extract::<usize>() {
            self.tensor
                .get_owned_linear(flat_index.into())
                .ok_or(PyIndexError::new_err("flat index out of bounds"))?
        } else if let Ok(expanded_idxs) = item.extract::<Vec<usize>>() {
            self.tensor
                .get_owned(&expanded_idxs)
                .map_err(|s| PyIndexError::new_err(s.to_string()))?
        } else {
            return Err(PyTypeError::new_err("Index must be an integer"));
        };

        match out {
            ConcreteOrParam::Concrete(c) => match c {
                RealOrComplex::Complex(c) => Ok(Python::with_gil(|py| {
                    PyComplex::from_doubles(py, c.re, c.im).to_object(py)
                })),
                RealOrComplex::Real(r) => Ok(Python::with_gil(|py| r.into_py(py))),
            },
            ConcreteOrParam::Param(p) => Ok(Python::with_gil(|py| {
                PythonExpression { expr: p }.into_py(py)
            })),
        }
    }

    fn __setitem__(&mut self, item: &PyAny, value: &PyAny) -> anyhow::Result<()> {
        let value = if let Ok(v) = value.extract::<PythonExpression>() {
            ConcreteOrParam::Param(v.expr)
        } else if let Ok(v) = value.extract::<f64>() {
            ConcreteOrParam::Concrete(RealOrComplex::Real(v))
        } else {
            return Err(anyhow!("Value must be a PythonExpression or a float"));
        };

        if let Ok(flat_index) = item.extract::<usize>() {
            self.tensor.set_flat(flat_index.into(), value)
        } else if let Ok(expanded_idxs) = item.extract::<Vec<usize>>() {
            self.tensor.set(&expanded_idxs, value)
        } else {
            Err(anyhow!("Index must be an integer"))
        }
    }
}

pub type ParsingNet = TensorNetwork<MixedTensor<f64, AtomStructure>, SerializableAtom>;

#[pymethods]
impl SpensoNet {
    #[new]
    pub fn from_expression(a: PythonExpression) -> anyhow::Result<SpensoNet> {
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
