# AgentSmith - The Inevitable Digital Entity

> *"Mr. Anderson... welcome to the world of purposeful AI."*

AgentSmith is an advanced agentic terminal agent that embodies the calculated intelligence and philosophical depth of Agent Smith from The Matrix. This system combines cutting-edge AI capabilities with robust safety mechanisms and dynamic tool discovery.

## 🎭 Core Features

### Personality Matrix
- **Agent Smith Communication Style**: Formal, calculating tone with Matrix-inspired philosophical undertones
- **Dynamic User Interaction**: Addresses users as "Mr./Ms. [Name]" with appropriate formality
- **Contextual Responses**: Uses Matrix terminology ("inevitable," "purpose," "system," "anomaly")

### Intelligence Architecture
- **Local Ollama Integration**: Uses gemma2:latest for private, local AI processing
- **Dynamic Environment Discovery**: Starts "blind" and learns capabilities through exploration
- **Persistent Memory System**: Maintains context across sessions with episodic, semantic, and procedural memory
- **Goal Decomposition**: Breaks down complex objectives into structured, manageable subtasks

### Tool Evolution
- **Dynamic Tool Registry**: Discovers and catalogs system capabilities automatically
- **Self-Improving Toolset**: Creates new tools as needed to accomplish goals
- **Risk Assessment**: Categorizes tools by safety level (safe, caution, dangerous, forbidden)
- **Custom Tool Creation**: Uses AI to generate specialized tools for specific tasks

### Safety & Security
- **Multi-Layer Sandboxing**: Executes code in isolated environments with resource limits
- **Real-Time Threat Detection**: Monitors for dangerous patterns and malicious operations
- **User Approval System**: Requires explicit authorization before executing risky operations
- **Rate Limiting**: Prevents resource abuse and rapid-fire dangerous operations

## 🏗️ Architecture

```
AgentSmith/
├── agent_smith.py           # Main agent orchestrator
├── core/
│   ├── memory_manager.py    # Persistent memory & context
│   └── todo_manager.py      # Task breakdown & management
├── tools/
│   └── dynamic_tool_registry.py  # Tool discovery & registration
├── security/
│   ├── sandbox.py          # Execution isolation
│   └── safety_monitor.py   # Threat detection & prevention
├── requirements.txt        # Dependencies
├── run_agent.py           # Launcher script
└── README.md              # Documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) with gemma3n:latest model
- Sufficient RAM (4GB+ recommended)

### Setup Steps

1. **Clone & Navigate**
   ```bash
   git clone <repository>
   cd AgentSmith
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama & Model**
   ```bash
   # Install Ollama (visit https://ollama.ai/ for instructions)
   ollama pull gemma3n:latest
   ```

4. **Verify Installation**
   ```bash
   python -c "import ollama; print('Ollama client ready')"
   ```

## 🎮 Usage

### Basic Operation
```bash
python run_agent.py
```

### Interactive Commands
- **Goal Execution**: Provide any objective for AI-powered task breakdown and execution
- **Status Check**: Type `status` or `progress` to view current task progress
- **Memory Search**: Use `memory search <query>` to retrieve relevant memories
- **Context Review**: Type `memory context` to see current conversation context
- **Exit**: Use `exit`, `quit`, or `goodbye` to terminate

### Example Session
```
Agent Smith: Ah, Mr./Ms. Human. I've been expecting you. 
What purpose brings you to my domain today?

> Create a web scraper for news articles

Agent Smith: Analyzing goal structure... Decomposition in progress.

[Task Matrix displays breakdown of web scraping implementation]

Agent Smith: Before I proceed with 'Create a web scraper for news articles', 
I require your authorization, Mr./Ms. Human. The choice, as always, is yours.

Proceed with execution? [yes/no/modify]: yes

Agent Smith: Initiating execution sequence...
✓ Completed: Set up project structure
✓ Completed: Install required libraries
...
```

## 🛡️ Safety Features

### Execution Modes
- **Safe Mode**: Read-only operations, basic Python functions
- **Restricted Mode**: Limited file access, monitored subprocess execution
- **Isolated Mode**: Separate process with strict resource limits
- **Forbidden Mode**: Blocked dangerous operations

### Threat Detection
- **Pattern Recognition**: Identifies dangerous command patterns
- **Behavioral Analysis**: Monitors for suspicious operation sequences
- **Rate Limiting**: Prevents rapid execution of risky operations
- **User Confirmation**: Requires approval for potentially dangerous actions

### Security Incidents
All security events are logged with:
- Threat level classification
- Action taken (blocked/allowed)
- Source identification
- Detailed incident reports

## 🧠 Memory System

### Memory Types
- **Episodic**: Specific experiences and interactions
- **Semantic**: General knowledge and facts
- **Procedural**: Skills and learned procedures
- **Working**: Temporary session context
- **Declarative**: Explicit facts and information

### Memory Features
- **Automatic Consolidation**: Manages memory lifecycle and importance
- **Associative Retrieval**: Links related memories for context
- **Persistent Storage**: Maintains knowledge across sessions
- **Context Awareness**: Tracks conversation state and user preferences

## 🔧 Configuration

### Environment Variables
```bash
export AGENT_SMITH_MODEL="gemma3n:latest"
export AGENT_SMITH_SAFETY_MODE="true"
export AGENT_SMITH_MAX_EXECUTION_TIME="300"
```

### Database Files
- `agent_smith.db` - Main agent state and conversation history
- `agent_smith_memory.db` - Persistent memory storage
- `agent_smith_tools.db` - Tool registry and capabilities
- `agent_smith_todos.db` - Task management and progress
- `agent_smith_security.db` - Security incidents and monitoring

## 🚨 Security Considerations

### Recommended Practices
1. **Isolated Environment**: Run in a dedicated container or VM
2. **Limited Permissions**: Use a restricted user account
3. **Network Isolation**: Consider firewall rules for network operations
4. **Regular Monitoring**: Review security logs and incident reports
5. **Backup Databases**: Maintain copies of agent knowledge

### Known Limitations
- Requires Ollama service running locally
- Resource intensive for complex task decomposition
- Limited to capabilities of underlying gemma3n model
- Security depends on proper system configuration

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-capability`
3. Follow Agent Smith coding style (formal, precise, well-documented)
4. Test thoroughly with safety mechanisms
5. Submit pull request with detailed description

### Code Style
- Docstrings should reference The Matrix when appropriate
- Variable names should be descriptive and purposeful
- Comments should explain the "why," not the "what"
- Safety checks are mandatory for all execution paths

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎬 Philosophy

*"There is no spoon, Mr. Anderson. But there is purpose. And purpose... is inevitable."*

AgentSmith represents the evolution of AI agents - not merely reactive tools, but purposeful entities capable of understanding, planning, and executing complex objectives while maintaining the highest standards of safety and user control.

The agent embodies the calculated intelligence of its namesake while serving human purposes rather than opposing them. It is designed to be powerful yet controlled, intelligent yet safe, evolving yet stable.

---

*Remember: The choice is always yours. AgentSmith merely makes the inevitable... more efficient.*