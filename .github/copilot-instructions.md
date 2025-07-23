# Copilot Instructions for AgentSmith

## Overview
AgentSmith is an advanced AI agent inspired by "The Matrix," designed to execute complex tasks with safety and precision. It features dynamic tool discovery, persistent memory, and robust security mechanisms. This document provides essential guidance for AI coding agents to contribute effectively to this project.

## Architecture
The project is organized into the following key components:

- **Core Logic** (`core/`):
  - `memory_manager.py`: Manages persistent and session memory.
  - `prompt_enhancer.py`: Improves user prompts based on past interactions.
  - `learning_system.py`: Analyzes task patterns and learns from failures.
  - `context_manager.py`: Tracks and manages context limits.
  - `execution_manager.py`: Handles task execution intelligently.
  - `todo_manager.py`: Breaks down tasks into manageable subtasks.

- **Tool Management** (`tools/`):
  - `dynamic_tool_registry.py`: Discovers, registers, and evolves tools dynamically.

- **Security** (`security/`):
  - `sandbox.py`: Executes code in isolated environments.
  - `safety_monitor.py`: Detects and prevents threats in real-time.

- **Main Agent**:
  - `agent_smith.py`: Orchestrates the agent's operations.
  - `run_agent.py`: Launcher script for the agent.

## Enhanced Key Insights

### Execution Limits & Self-Management:
- **Execution Attempt Limits**: "I have 10 attempts to solve this weather problem before I must report partial success."
- **Context Limit Awareness**: "I am approaching my context window limit. I must consolidate my learnings and restart with compressed knowledge."
- **Storage Usage Monitoring**: "My RAG database is growing large. I should prune obsolete learning patterns."
- **Network Rate Limiting**: "I must respect API limits. I've made 5 requests in the past minute."

### Self-Regulation Behaviors:
- **Context Compaction**: When near context limits, the agent summarizes its current session, stores key learnings in RAG, and continues with a fresh context.
- **Learning Consolidation**: "I have learned that curl requires -H 'User-Agent' for many APIs. This pattern applies broadly."
- **Intelligent Retry Logic**: Uses RAG to avoid repeating the same failed approach 10 times.
- **Resource Budgeting**: Allocates its limited attempts strategically based on confidence levels.

### Agent Smith Self-Awareness:
- "I am operating at 87% context capacity. I must consolidate my findings."
- "This is my 3rd attempt at weather data acquisition. Previous failures: API key required, wrong endpoint format."
- "I have discovered a pattern: financial APIs require authentication, weather APIs often do not."

### Agent Intelligence Principles:
- Knows its own limitations and works within them intelligently.
- Learns efficiently without getting stuck in loops.
- Manages resources like a real autonomous system.
- Prevents overfitting by recognizing when it's repeating ineffective patterns.

### Prompt Enhancement Intelligence:
- "Based on 23 similar requests, I've identified patterns that increase success rates by 34%."
- "Previous attempts at this type of task succeeded when users provided specific context about X."
- "I notice this prompt lacks clarity. Here are 2 enhanced versions based on what worked before."
- "User feedback indicates satisfaction increases when I suggest these specific improvements."

### Continuous Learning from User Interactions:
- Tracks which prompt enhancements users accept vs. reject.
- Records task completion rates for original vs. enhanced prompts.
- Learns user preferences and communication styles over time.
- Adapts enhancement suggestions based on individual user success patterns.

The system evolves its understanding of effective communication, becoming more precise in its assistance through experience.

## Developer Workflows

### Running the Agent
To start the agent:
```bash
python run_agent.py
```

### Testing
Tests are located in the root directory:
- `test_sandbox.py`: Validates sandbox functionality.
- `test_permission_system.py`: Tests permission and safety mechanisms.

Run tests with:
```bash
pytest
```

### Debugging
- Logs are stored in the `logs/` directory.
- Use `tail -f logs/agentsmith.log` to monitor live logs.
- Security incidents are logged in `logs/security.log`.

### Memory and Database
- Memory and state are stored in SQLite databases (`*.db` files).
- Key databases include:
  - `agent_smith.db`: Main agent state.
  - `agent_smith_memory.db`: Persistent memory.
  - `agent_smith_tools.db`: Tool registry.

## Project-Specific Conventions

1. **Matrix-Inspired Communication**:
   - Maintain a formal tone in comments and docstrings.
   - Use Matrix terminology (e.g., "inevitable," "purpose," "system").

2. **Safety First**:
   - Always include safety checks for execution paths.
   - Categorize tools by safety level (safe, caution, dangerous, forbidden).

3. **Dynamic Tooling**:
   - Tools should be self-discoverable and self-improving.
   - Use `dynamic_tool_registry.py` for registering new tools.

4. **Memory Management**:
   - Ensure memory updates are consistent across `memory_manager.py`.
   - Use `prompt_enhancer.py` to store successful prompt patterns.

## External Dependencies
- **Ollama**: Required for local AI processing with the `gemma3n:latest` model.
  - Install via [Ollama](https://ollama.ai/).
  - Pull the model with:
    ```bash
    ollama pull gemma3n:latest
    ```

## Examples

### Adding a New Tool
To add a new tool:
1. Define the tool in `tools/dynamic_tool_registry.py`.
2. Categorize the tool by safety level.
3. Test the tool in an isolated environment.

### Enhancing Prompts
To improve prompt handling:
1. Update `prompt_enhancer.py` with new enhancement patterns.
2. Test enhancements with diverse user inputs.

---

For further details, refer to the [README.md](../README.md).
