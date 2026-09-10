use agentwerk::Condition;
use pyo3::prelude::*;

use crate::agent::PyAgent;
use crate::query::value_error;
use crate::task::to_task;

/// Release agents and tasks once per run when an AQL query matches.
#[pyclass(name = "Condition")]
pub struct PyCondition {
    inner: Option<Condition>,
}

impl PyCondition {
    fn set(&mut self, edit: impl FnOnce(Condition) -> Condition) {
        let condition = self.inner.take().expect("a setter kept the condition");
        self.inner = Some(edit(condition));
    }

    pub(crate) fn get(&self) -> &Condition {
        self.inner.as_ref().expect("a setter kept the condition")
    }
}

#[pymethods]
impl PyCondition {
    /// Compile AQL as the query that releases this condition's work.
    #[new]
    fn new(aql: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Some(Condition::new(aql).map_err(|error| value_error(error.to_string()))?),
        })
    }

    /// Set the runtime identity, replacing one already set.
    fn id(mut slf: PyRefMut<'_, Self>, id: String) -> PyRefMut<'_, Self> {
        slf.set(|condition| condition.id(id));
        slf
    }

    /// Add an agent to activate when the condition matches.
    fn add_agent<'py>(
        mut slf: PyRefMut<'py, Self>,
        agent: PyRef<'_, PyAgent>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let agent = agent.ready()?.clone();
        slf.set(|condition| condition.add_agent(agent));
        Ok(slf)
    }

    /// Add a task to create when the condition matches.
    fn add_task<'py>(
        mut slf: PyRefMut<'py, Self>,
        task: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let task = to_task(task)?;
        slf.set(|condition| condition.add_task(task));
        Ok(slf)
    }
}
