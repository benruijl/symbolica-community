use std::collections::HashMap;
use std::env;

use pyo3::types::PyType;
use pyo3::{exceptions, pyclass, pymethods, FromPyObject, PyErr, PyRef, Python};
use pyo3::{PyObject, PyResult};
use symbolica::api::python::PythonExpression;
use symbolica::atom::Atom;
use symbolica::domains::float::{Complex, Float, RealNumberLike};
use symbolica::state::State;
use vakint::{
    EvaluationMethod, EvaluationOrder, FMFTOptions, LoopNormalizationFactor, MATADOptions,
    NumericalEvaluationResult, PySecDecOptions, Vakint, VakintError, VakintExpression,
    VakintSettings,
};

fn vakint_to_python_error(vakint_error: VakintError) -> PyErr {
    exceptions::PyValueError::new_err(format!("Vakint error | {vakint_error}"))
}

#[pyclass(name = "Vakint", module = "symbolica", subclass)]
pub struct VakintWrapper {
    pub vakint: Vakint,
}

#[pyclass(name = "VakintNumericalResult", module = "symbolica", subclass)]
pub struct NumericalEvaluationResultWrapper {
    pub value: NumericalEvaluationResult,
}

impl<'a> FromPyObject<'a> for NumericalEvaluationResultWrapper {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<NumericalEvaluationResultWrapper>() {
            Ok(NumericalEvaluationResultWrapper { value: a.value })
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid vakint numerical resuls (Laurent series in epsilon)",
            ))
        }
    }
}

#[pymethods]
impl NumericalEvaluationResultWrapper {
    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.value))
    }

    pub fn to_list(&self) -> PyResult<Vec<(i64, (f64, f64))>> {
        Ok(self
            .value
            .0
            .iter()
            .map(|(exp, coeff)| (*exp, (coeff.re.to_f64(), coeff.im.to_f64())))
            .collect())
    }
    #[new]
    pub fn __new__(values: Vec<(i64, (f64, f64))>) -> PyResult<NumericalEvaluationResultWrapper> {
        let r = NumericalEvaluationResult(
            values
                .iter()
                .map(|(eps_pwr, (re, im))| {
                    (
                        *eps_pwr,
                        Complex::new(Float::with_val(53, re), Float::with_val(53, im)),
                    )
                })
                .collect::<Vec<_>>(),
        );
        Ok(NumericalEvaluationResultWrapper { value: r })
    }

    pub fn compare_to(
        &self,
        other: PyRef<NumericalEvaluationResultWrapper>,
        relative_threshold: f64,
        error: Option<PyRef<NumericalEvaluationResultWrapper>>,
        max_pull: Option<f64>,
    ) -> PyResult<(bool, String)> {
        Ok(self.value.does_approx_match(
            &other.value,
            error.map(|e| e.value.clone()).as_ref(),
            relative_threshold,
            max_pull.unwrap_or(3.0),
        ))
    }
}

#[pyclass(name = "VakintExpression", module = "symbolica", subclass)]
pub struct VakintExpressionWrapper {
    pub value: VakintExpression,
}

impl<'a> FromPyObject<'a> for VakintExpressionWrapper {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<VakintExpressionWrapper>() {
            Ok(VakintExpressionWrapper { value: a.value })
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid vakint expression",
            ))
        }
    }
}

#[pymethods]
impl VakintExpressionWrapper {
    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.value))
    }

    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        let a: Atom = self.value.clone().into();
        Ok(a.into())
    }

    #[new]
    pub fn new(atom: PyObject) -> PyResult<VakintExpressionWrapper> {
        let a = Python::with_gil(|py| atom.extract::<PythonExpression>(py))?;
        Ok(VakintExpressionWrapper {
            value: VakintExpression::try_from(a.expr).map_err(vakint_to_python_error)?,
        })
    }
}

#[pyclass(name = "VakintEvaluationMethod", module = "symbolica", subclass)]
pub struct VakintEvaluationMethodWrapper {
    pub method: EvaluationMethod,
}

impl<'a> FromPyObject<'a> for VakintEvaluationMethodWrapper {
    fn extract(ob: &'a pyo3::PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<VakintEvaluationMethodWrapper>() {
            Ok(VakintEvaluationMethodWrapper { method: a.method })
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid vakint evaluation method",
            ))
        }
    }
}

#[pymethods]
impl VakintEvaluationMethodWrapper {
    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.method))
    }

    #[classmethod]
    pub fn new_alphaloop_method(_cls: &PyType) -> PyResult<VakintEvaluationMethodWrapper> {
        Ok(VakintEvaluationMethodWrapper {
            method: EvaluationMethod::AlphaLoop,
        })
    }

    #[classmethod]
    pub fn new_matad_method(
        _cls: &PyType,
        expand_masters: Option<bool>,
        susbstitute_masters: Option<bool>,
        substitute_hpls: Option<bool>,
        direct_numerical_substition: Option<bool>,
    ) -> PyResult<VakintEvaluationMethodWrapper> {
        Ok(VakintEvaluationMethodWrapper {
            method: EvaluationMethod::MATAD(MATADOptions {
                expand_masters: expand_masters.unwrap_or(true),
                susbstitute_masters: susbstitute_masters.unwrap_or(true),
                substitute_hpls: substitute_hpls.unwrap_or(true),
                direct_numerical_substition: direct_numerical_substition.unwrap_or(true),
            }),
        })
    }

    #[classmethod]
    pub fn new_fmft_method(
        _cls: &PyType,
        expand_masters: Option<bool>,
        susbstitute_masters: Option<bool>,
    ) -> PyResult<VakintEvaluationMethodWrapper> {
        Ok(VakintEvaluationMethodWrapper {
            method: EvaluationMethod::FMFT(FMFTOptions {
                expand_masters: expand_masters.unwrap_or(true),
                susbstitute_masters: susbstitute_masters.unwrap_or(true),
            }),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    pub fn new_pysecdec_method(
        _cls: &PyType,
        quiet: Option<bool>,
        relative_precision: Option<f64>,
        min_n_evals: Option<u64>,
        max_n_evals: Option<u64>,
        reuse_existing_output: Option<String>,
        numerical_masses: Option<HashMap<String, f64>>,
        numerical_external_momenta: Option<HashMap<usize, (f64, f64, f64, f64)>>,
    ) -> PyResult<VakintEvaluationMethodWrapper> {
        let ext_mom = if let Some(em) = numerical_external_momenta {
            HashMap::from_iter(em.iter().map(|(i_ext, ps)| (format!("p{}", i_ext), *ps)))
        } else {
            HashMap::new()
        };
        Ok(VakintEvaluationMethodWrapper {
            method: EvaluationMethod::PySecDec(PySecDecOptions {
                quiet: quiet.unwrap_or(true),
                reuse_existing_output,
                relative_precision: relative_precision.unwrap_or(1e-7),
                min_n_evals: min_n_evals.unwrap_or(10_000),
                max_n_evals: max_n_evals.unwrap_or(1_000_000_000_000),
                numerical_masses: numerical_masses.unwrap_or_default(),
                numerical_external_momenta: ext_mom,
            }),
        })
    }
}

#[pymethods]
impl VakintWrapper {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        run_time_decimal_precision: Option<u32>,
        evaluation_order: Option<Vec<PyRef<VakintEvaluationMethodWrapper>>>,
        epsilon_symbol: Option<String>,
        mu_r_sq_symbol: Option<String>,
        form_exe_path: Option<String>,
        python_exe_path: Option<String>,
        verify_numerator_identification: Option<bool>,
        integral_normalization_factor: Option<String>,
        allow_unknown_integrals: Option<bool>,
        clean_tmp_dir: Option<bool>,
        number_of_terms_in_epsilon_expansion: Option<i64>,
        use_dot_product_notation: Option<bool>,
        temporary_directory: Option<String>,
    ) -> PyResult<VakintWrapper> {
        let norm_factor = if let Some(nf) = integral_normalization_factor {
            match nf.as_str() {
                "MSbar" => LoopNormalizationFactor::MSbar,
                "pySecDec" => LoopNormalizationFactor::pySecDec,
                "FMFTandMATAD" => LoopNormalizationFactor::FMFTandMATAD,
                s => LoopNormalizationFactor::Custom(s.into()),
            }
        } else {
            LoopNormalizationFactor::MSbar
        };
        // let eval_order = if let Some(eo) = evaluation_order {
        //     let mut eval_stack = vec![];
        //     for m in eo.iter() {
        //         let method = Python::with_gil(|py| m.extract::<VakintEvaluationMethodWrapper>(py))?;
        //         eval_stack.push(method.method);
        //     }
        //     EvaluationOrder(eval_stack)
        // } else {
        //     EvaluationOrder::all()
        // };
        let eval_order = if let Some(eo) = evaluation_order {
            EvaluationOrder(
                eo.iter()
                    .map(|m| m.method.clone())
                    .collect::<Vec<EvaluationMethod>>(),
            )
        } else {
            EvaluationOrder::all()
        };
        #[allow(clippy::needless_update)]
        let vakint = Vakint::new(Some(VakintSettings {
            run_time_decimal_precision: run_time_decimal_precision.unwrap_or(17),
            epsilon_symbol: epsilon_symbol.unwrap_or("ε".into()),
            mu_r_sq_symbol: mu_r_sq_symbol.unwrap_or("mursq".into()),
            form_exe_path: form_exe_path.unwrap_or("form".into()),
            python_exe_path: python_exe_path.unwrap_or("python3".into()),
            verify_numerator_identification: verify_numerator_identification.unwrap_or(true),
            integral_normalization_factor: norm_factor,
            allow_unknown_integrals: allow_unknown_integrals.unwrap_or(true),
            clean_tmp_dir: clean_tmp_dir.unwrap_or(env::var("VAKINT_NO_CLEAN_TMP_DIR").is_err()),
            temporary_directory,
            evaluation_order: eval_order,
            // This quantity is typically set equal to *one plus the maximum loop count* of the UV regularisation problem considered.
            // For example when considering a 2-loop problem, then:
            //   a) for the nested one-loop integrals appearing, the single pole, finite term *and* order-epsilon term will need to be considered.
            //   b) for the two-loop integrals, the double pole, single pole and finite terms will be needed, so again three terms
            number_of_terms_in_epsilon_expansion: number_of_terms_in_epsilon_expansion.unwrap_or(4),
            use_dot_product_notation: use_dot_product_notation.unwrap_or(false),
            ..VakintSettings::default()
        }))
        .map_err(vakint_to_python_error)?;
        let wrapper = VakintWrapper { vakint };
        Ok(wrapper)
    }

    pub fn numerical_result_from_expression(
        &self,
        expr: PythonExpression,
    ) -> PyResult<NumericalEvaluationResultWrapper> {
        let value = NumericalEvaluationResult::from_atom(
            expr.expr.as_view(),
            State::get_symbol(self.vakint.settings.epsilon_symbol.clone()),
            &self.vakint.settings,
        )
        .map_err(vakint_to_python_error)?;
        Ok(NumericalEvaluationResultWrapper { value })
    }

    pub fn numerical_evaluation(
        &self,
        evaluated_integral: PyObject,
        params: HashMap<String, f64>,
        externals: Option<HashMap<usize, (f64, f64, f64, f64)>>,
    ) -> PyResult<(
        NumericalEvaluationResultWrapper,
        Option<NumericalEvaluationResultWrapper>,
    )> {
        let evaluated_integral_atom =
            Python::with_gil(|py| evaluated_integral.extract::<PythonExpression>(py))?;
        let p = self.vakint.params_from_f64(&params);
        let e = externals.map(|ext| self.vakint.externals_from_f64(&ext));
        let (numerical_result, numerical_error) = self
            .vakint
            .numerical_evaluation(evaluated_integral_atom.expr.as_view(), &p, e.as_ref())
            .map_err(vakint_to_python_error)?;
        Ok((
            NumericalEvaluationResultWrapper {
                value: numerical_result,
            },
            numerical_error.map(|ne| NumericalEvaluationResultWrapper { value: ne }),
        ))
    }

    pub fn numerical_result_to_expression(
        &self,
        res: PyRef<NumericalEvaluationResultWrapper>,
    ) -> PyResult<PythonExpression> {
        let value = res.value.to_atom(State::get_symbol(
            self.vakint.settings.epsilon_symbol.clone(),
        ));
        Ok(value.into())
    }

    pub fn to_canonical(
        &self,
        a: PythonExpression,
        short_form: Option<bool>,
    ) -> PyResult<PythonExpression> {
        let result = self
            .vakint
            .to_canonical(a.expr.as_view(), short_form.unwrap_or(false))
            .map_err(vakint_to_python_error)?;
        Ok(result.into())
    }

    pub fn tensor_reduce(&self, a: PythonExpression) -> PyResult<PythonExpression> {
        let result = self
            .vakint
            .tensor_reduce(a.expr.as_view())
            .map_err(vakint_to_python_error)?;
        Ok(result.into())
    }

    pub fn evaluate_integral(&self, a: PythonExpression) -> PyResult<PythonExpression> {
        let result = self
            .vakint
            .evaluate_integral(a.expr.as_view())
            .map_err(vakint_to_python_error)?;
        Ok(result.into())
    }

    pub fn evaluate(&self, a: PythonExpression) -> PyResult<PythonExpression> {
        let result = self
            .vakint
            .evaluate(a.expr.as_view())
            .map_err(vakint_to_python_error)?;
        Ok(result.into())
    }
}
