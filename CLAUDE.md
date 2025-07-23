# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Run the agent:**
```bash
python run_agent.py
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Database management:**
The system creates several SQLite databases:
- `agent_smith.db` - Main agent state and conversation history
- `agent_smith_memory.db` - Persistent memory storage and prompt experiences
- `agent_smith_learnings.db` - Task learning patterns and failure analysis
- `agent_smith_tools.db` - Tool registry and capabilities
- `agent_smith_todos.db` - Task management and progress
- `agent_smith_security.db` - Security incidents and monitoring
- `agent_smith_users.db` - User preferences and designations

**Testing:**
No formal test framework configured. Manual testing by running the agent and observing behavior.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) with gemma3n:latest model running locally
- Dependencies from requirements.txt (langgraph, langchain-core, ollama, psutil, rich, etc.)

## Architecture Overview

**Core Philosophy:**
AgentSmith is an autonomous agentic system that embodies Agent Smith's personality from The Matrix while providing AI assistance. The system is built around safety, memory persistence, and dynamic capability discovery.

**Main Components:**

1. **Agent Core** (`agent_smith.py`):
   - Main orchestrator using LangGraph for state management
   - Agent Smith personality with Matrix-inspired dialogue
   - Goal decomposition and execution workflow
   - Integrates all subsystems

2. **Memory System** (`core/memory_manager.py`):
   - Persistent episodic, semantic, and procedural memory
   - Memory consolidation and importance weighting
   - Conversation context tracking
   - SQLite-based storage with associative retrieval

3. **Task Management** (`core/todo_manager.py`):
   - AI-powered goal decomposition into subtasks
   - Hierarchical task tracking with dependencies
   - Progress monitoring and completion detection
   - Database persistence of task states

4. **Security Layer** (`security/`):
   - **Sandbox** (`sandbox.py`): Multi-level execution isolation (safe/restricted/isolated/forbidden)
   - **Safety Monitor** (`safety_monitor.py`): Pattern-based threat detection, rate limiting, incident logging

5. **Dynamic Tools** (`tools/dynamic_tool_registry.py`):
   - Runtime capability discovery (CLI tools, Python modules, filesystem ops)
   - Risk assessment and tool validation
   - AI-generated custom tools for specific goals
   - Persistent tool registry

6. **Prompt Enhancement System** (`core/prompt_enhancer.py`):
   - Analyzes user prompts based on previous experience
   - Stores prompt outcomes and success patterns
   - Generates improved prompt suggestions using Ollama
   - Learns from user preferences and task completion rates
   - Integrates seamlessly with memory and learning systems

**Execution Flow:**
1. **Perceive**: Discover environment and capabilities
2. **Enhance Prompt**: Analyze and potentially improve user prompts based on past experience
3. **Analyze**: Use Ollama/gemma3n to analyze user goals
4. **Plan**: Break goals into structured subtasks
5. **Approve**: Request user authorization before execution
6. **Execute**: Run tasks through sandboxed execution
7. **Reflect**: Learn from results and adapt, recording prompt outcomes for future enhancement

**Key Design Patterns:**
- All dangerous operations require explicit user approval
- Multi-layer security with pattern matching and sandboxing
- Persistent memory across sessions
- Dynamic tool discovery and creation
- Agent Smith personality integration throughout
- Continuous prompt improvement based on experience and outcomes
- User-guided enhancement selection with transparent reasoning

**Safety Architecture:**
- Sandboxed execution with resource limits
- Real-time threat pattern detection
- Rate limiting and abuse prevention
- Comprehensive security incident logging
- User approval gates for risky operations

## Configuration

Set environment variables for customization:
```bash
export AGENT_SMITH_MODEL="gemma3n:latest"
export AGENT_SMITH_SAFETY_MODE="true"
export AGENT_SMITH_MAX_EXECUTION_TIME="300"
```

## Working with the Codebase

**Adding new capabilities:**
- New tools: Extend `DynamicToolRegistry` or let the agent create them
- New memory types: Extend `MemoryType` enum in `memory_manager.py`
- New security patterns: Add to SafetyMonitor's pattern lists
- New execution modes: Extend `ExecutionMode` in `sandbox.py`
- New prompt categories: Extend `_categorize_prompt()` in `prompt_enhancer.py`
- New enhancement patterns: Modify analysis prompts in `PromptEnhancer` class

**Database schemas:**
Each component maintains its own SQLite database with proper foreign key relationships and indexing for performance.

**Agent personality:**
The Agent Smith persona is integrated throughout the system via personality prompts and formal communication style. All user-facing messages should maintain this character.

## Model Configuration

- **Default Model**: Gemma 3n (gemma3n:latest) is the default model for the project

## Design Principles

- **Flexible Intelligence**:
  - Do not hardcode any tool implementations or specific instructions instead allowing the model to figure out what needs to be done.

## Enhanced Key Insights

### Execution Limits & Self-Management:
- Execution Attempt Limits: "I have 10 attempts to solve this weather problem before I must report partial success"
- Context Limit Awareness: "I am approaching my context window limit. I must consolidate my learnings and restart with compressed knowledge"
- Storage Usage Monitoring: "My RAG database is growing large. I should prune obsolete learning patterns"
- Network Rate Limiting: "I must respect API limits. I've made 5 requests in the past minute"

### Self-Regulation Behaviors:
- Context Compaction: When near context limits, the agent summarizes its current session, stores key learnings in RAG, and continues with a fresh context
- Learning Consolidation: "I have learned that curl requires -H 'User-Agent' for many APIs. This pattern applies broadly."
- Intelligent Retry Logic: Uses RAG to avoid repeating the same failed approach 10 times
- Resource Budgeting: Allocates its limited attempts strategically based on confidence levels

### Agent Smith Self-Awareness:
- "I am operating at 87% context capacity. I must consolidate my findings."
- "This is my 3rd attempt at weather data acquisition. Previous failures: API key required, wrong endpoint format."
- "I have discovered a pattern: financial APIs require authentication, weather APIs often do not."

### Agent Intelligence Principles:
- Knows its own limitations and works within them intelligently
- Learns efficiently without getting stuck in loops
- Manages resources like a real autonomous system
- Prevents overfitting by recognizing when it's repeating ineffective patterns

The agent becomes truly intelligent about its own cognitive process, not just the external tasks.

### Prompt Enhancement Intelligence:
- "Based on 23 similar requests, I've identified patterns that increase success rates by 34%"
- "Previous attempts at this type of task succeeded when users provided specific context about X"
- "I notice this prompt lacks clarity. Here are 2 enhanced versions based on what worked before"
- "User feedback indicates satisfaction increases when I suggest these specific improvements"

### Continuous Learning from User Interactions:
- Tracks which prompt enhancements users accept vs. reject
- Records task completion rates for original vs. enhanced prompts
- Learns user preferences and communication styles over time
- Adapts enhancement suggestions based on individual user success patterns

The system evolves its understanding of effective communication, becoming more precise in its assistance through experience.