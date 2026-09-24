# Agents

Create an agent with a role, a model, and the tools it can call.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/agents.gif" width="600" alt="An agent processing tasks" />

```python
from agentwerk import Agent, ReadFileTool

agent = (
    Agent.from_env()
    .role("You are a release manager who prepares release notes.")
    .tool(ReadFileTool())
)

agent.add_task("Read CHANGELOG.md and summarize the entries added since the last release.")

results = await agent.finish()
```

## Construct an agent

```python
agent = Agent()
```

## Configure an agent

Use these members to define the agent's role, tools, routing, and shared context.

| Member | Purpose |
| --- | --- |
| `role(role)` | Define who the agent is and how it should work. |
| `tool(tool)` | Register a tool the agent may call. |
| `tools(tools)` | Register several tools together. |
| `label(label)` | Restrict the agent to tasks carrying this label. |
| `dir(dir)` | Set the directory the agent can access. |
| `template(key, value)` | Set a shared template value. |
| `templates(variables)` | Set several shared template values together. |
| `knowledge(store)` | Share a knowledge store and register its `KnowledgeTool`. |
| `interactive()` | Keep a task in progress while waiting for new instructions. |

## Run an agent

Use these members to submit work, start execution, and collect results.

| Member | Purpose |
| --- | --- |
| `add_task(task)` | Submit a task and return its ID. |
| `start()` | Process tasks in the background. |
| `finish_task(query)` | Wait for all matches and return the first result in query order. |
| `finish_tasks(query)` | Wait for matching tasks and return their results. |
| `finish()` | Run tasks and return their results. |
| `get_id()` | Get the agent's unique identifier. |

## Providers

Send an agent's model requests to Anthropic, OpenAI, Mistral, or a LiteLLM proxy.

```python
from agentwerk import Agent, Anthropic

provider = Anthropic(key)
agent = (
    Agent()
    .provider(provider)
    .model("claude-sonnet-4-20250514")
)
```

Load the provider or model separately from environment variables with `.provider(Provider.from_env())` or `.model(Model.from_env())`. Claude, GPT, Mistral, and Qwen families have built-in context-window and reasoning settings; override them or configure a custom model when needed:

```python
from agentwerk import Agent, Model

model = (
    Model("my-local-model")
    .context_window(128_000)
    .reasoning_effort("high")
)

agent = Agent().model(model)
```

### Provider configuration

Use these members to select, configure, and verify a provider.

| Member | Purpose |
| --- | --- |
| `provider(provider)` | Set the LLM provider. |
| `model(model)` | Set the model. |
| `Agent.from_env()` | Read the provider and model from environment variables. |
| `verify(model)` | Verify that the provider can answer with a model. |
| `Anthropic(key, base_url=..., timeout=...)` | Configure Anthropic. |
| `OpenAi(key, base_url=..., timeout=...)` | Configure OpenAI. |
| `Mistral(key, base_url=..., timeout=...)` | Configure Mistral. |
| `LiteLlm(key, base_url=..., timeout=...)` | Configure LiteLLM. |

### Provider environment

Set `LITELLM_PROVIDER` to choose a provider explicitly. Otherwise, API keys are checked in the order shown below.

| Variable | Purpose |
| --- | --- |
| `LITELLM_PROVIDER` | Choose `anthropic`, `mistral`, `openai`, or `litellm` outright, ahead of the keys below. |
| `LITELLM_API_KEY` | Authenticate with LiteLLM. |
| `MISTRAL_API_KEY` | Authenticate with Mistral. |
| `ANTHROPIC_API_KEY` | Authenticate with Anthropic. |
| `OPENAI_API_KEY` | Authenticate with OpenAI. |
| `LITELLM_BASE_URL` | Set a different LiteLLM API address. |
| `MISTRAL_BASE_URL` | Set a different Mistral API address. |
| `ANTHROPIC_BASE_URL` | Set a different Anthropic API address. |
| `OPENAI_BASE_URL` | Set a different OpenAI API address. |
| `SSL_CERT_FILE` | Trust the CA certificates in this file instead of the built-in root store. |
| `SSL_CERT_DIR` | Trust the CA certificates in this directory instead of the built-in root store. |

### Model configuration

Use these members to configure and inspect model limits.

| Member | Purpose |
| --- | --- |
| `context_window(size)` | Set the context window size for a model. |
| `get_context_window()` | Get the configured window size. |
| `reasoning_effort(effort)` | Set the reasoning level. |
| `get_reasoning_effort()` | Get the configured effort. |

### Model environment

Set `MODEL` to override provider-specific model variables.

| Variable | Purpose |
| --- | --- |
| `MODEL` | Set the model name. |
| `ANTHROPIC_MODEL` | Set the Anthropic model when `MODEL` is unset. |
| `OPENAI_MODEL` | Set the OpenAI model when `MODEL` is unset. |
| `MISTRAL_MODEL` | Set the Mistral model when `MODEL` is unset. |
| `LITELLM_MODEL` | Set the LiteLLM model when `MODEL` is unset. |
| `MODEL_CONTEXT_WINDOW` | Set the context window size in tokens. |

## Interactive agents

Call `interactive()` to keep a task open for follow-up replies. Interactive agents have no completion tool by default.

```python
agent = Agent.from_env().interactive()
id = agent.add_task("Where does the configuration get loaded?")

werk = agent.start()
await werk.finish()

werk.add_reply(id, "And which environment variables override it?")
await werk.finish()
```

Replies pause the task in `in_progress`, and completion methods return when it pauses. Use `add_reply(id, content)` to resume and `set_task_finished(id, result)` to end the conversation. Intermediate replies arrive as [events](events.md). `on_result` receives the final result.
