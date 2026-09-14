//! Exposes task, event, and joined AQL queries through one Python class.

use agentwerk::agents::Matcher;
use agentwerk::event::Event;
use agentwerk::{Query, Task};
use pyo3::prelude::*;

use crate::event::to_py_event;
use crate::task::PyTask;

/// Selects tasks, events, or joined task-event rows by qualified field values.
#[pyclass(name = "Query")]
pub struct PyQuery {
    source: String,
    query: Query,
}

#[pymethods]
impl PyQuery {
    /// Compile an AQL string, the same syntax a string argument carries.
    ///
    #[new]
    fn new(query: &str) -> PyResult<Self> {
        Ok(PyQuery {
            source: query.to_string(),
            query: compile_query(query)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Query({:?})", self.source)
    }
}

pub(crate) fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn compile_query(query: &str) -> PyResult<Query> {
    Query::new(query).map_err(|error| value_error(error.to_string()))
}

/// Read a compiled query or an AQL string without accepting a callable.
pub(crate) fn to_query(arg: &Bound<'_, PyAny>) -> PyResult<Query> {
    if let Ok(query) = arg.extract::<PyRef<'_, PyQuery>>() {
        return Ok(query.query.clone());
    }
    if let Ok(query) = arg.extract::<String>() {
        return compile_query(&query);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a Query or AQL string",
    ))
}

/// Read a Python argument for an operation that ultimately selects tasks: a
/// `Query`, a string in AQL, or a callable as a condition of its own. Named
/// AQL may originate from tasks, events, or their join; callables receive a
/// task. A string that does not compile raises `ValueError`.
pub fn to_task_matcher(py: Python<'_>, arg: &Py<PyAny>) -> PyResult<Query> {
    if let Ok(query) = arg.extract::<PyRef<'_, PyQuery>>(py) {
        return Ok(query.query.clone());
    }
    if let Ok(query) = arg.extract::<String>(py) {
        return compile_query(&query);
    }
    let callable = arg.clone_ref(py);
    Ok(Matcher::into_query(move |task: &Task| {
        task_predicate(&callable, task)
    }))
}

/// Read an event finder's query. Named AQL may originate from tasks or events;
/// callables continue to receive the destination event.
pub fn to_event_matcher(py: Python<'_>, arg: &Py<PyAny>) -> PyResult<Query> {
    if let Ok(query) = arg.extract::<PyRef<'_, PyQuery>>(py) {
        return Ok(query.query.clone());
    }
    if let Ok(query) = arg.extract::<String>(py) {
        return compile_query(&query);
    }
    let callable = arg.clone_ref(py);
    Ok(Matcher::into_query(move |event: &Event| {
        event_predicate(&callable, event)
    }))
}

/// Ask a Python condition about a task. A conversion or Python error reads as
/// false, so a broken condition never brings down an agent's thread.
fn task_predicate(predicate: &Py<PyAny>, task: &Task) -> bool {
    Python::attach(|py| {
        Py::new(py, PyTask::from_task(task))
            .and_then(|view| predicate.bind(py).call1((view,)))
            .and_then(|value| value.is_truthy())
            .unwrap_or(false)
    })
}

/// Ask a Python condition about an event. A Python error reads as false, so a
/// broken condition never stops the read.
fn event_predicate(predicate: &Py<PyAny>, event: &Event) -> bool {
    Python::attach(|py| {
        predicate
            .bind(py)
            .call1((to_py_event(event),))
            .and_then(|value| value.is_truthy())
            .unwrap_or(false)
    })
}
