use agentwerk::Condition;
use pyo3::prelude::*;

use crate::agent::PyAgent;
use crate::query::to_query;
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
    /// Create a condition released by an AQL string or compiled query.
    #[new]
    fn new(query: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Some(Condition::new(to_query(query)?)),
        })
    }

    /// Add an agent to activate when the condition matches.
    fn agent<'py>(
        mut slf: PyRefMut<'py, Self>,
        agent: PyRef<'_, PyAgent>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let agent = agent.ready()?.clone();
        slf.set(|condition| condition.agent(agent));
        Ok(slf)
    }

    /// Add agents to activate when the condition matches.
    fn agents<'py>(
        mut slf: PyRefMut<'py, Self>,
        agents: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let agents = agents
            .try_iter()?
            .map(|agent| {
                let agent = agent?;
                let agent = agent.extract::<PyRef<'_, PyAgent>>()?;
                Ok(agent.ready()?.clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        slf.set(|condition| condition.agents(agents));
        Ok(slf)
    }

    /// Add a task to create when the condition matches.
    fn task<'py>(
        mut slf: PyRefMut<'py, Self>,
        task: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let task = to_task(task)?;
        slf.set(|condition| condition.task(task));
        Ok(slf)
    }

    /// Add tasks to create when the condition matches.
    fn tasks<'py>(
        mut slf: PyRefMut<'py, Self>,
        tasks: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let tasks = tasks
            .try_iter()?
            .map(|task| to_task(&task?))
            .collect::<PyResult<Vec<_>>>()?;
        slf.set(|condition| condition.tasks(tasks));
        Ok(slf)
    }
}
