# AgentSmith - The Inevitable Digital Entity

> *"Mr. Anderson... welcome to the world of purposeful AI."*

AgentSmith is an advanced agentic terminal agent that embodies the calculated intelligence and philosophical depth of Agent Smith from The Matrix. This system combines cutting-edge AI capabilities with robust safety mechanisms and dynamic tool discovery.

## 🎭 Core Features

### Personality Matrix
- **Agent Smith Communication Style**: Formal, calculating tone with Matrix-inspired philosophical undertones
- **Dynamic User Interaction**: Addresses users as "Mr./Ms. [Name]" with appropriate formality
- **Contextual Responses**: Uses Matrix terminology ("inevitable," "purpose," "system," "anomaly")

### Intelligence Architecture
- **Local Ollama Integration**: Uses gemma3n:latest for private, local AI processing
- **Dynamic Environment Discovery**: Starts "blind" and learns capabilities through exploration
- **Persistent Memory System**: Maintains context across sessions with episodic, semantic, and procedural memory
- **Goal Decomposition**: Breaks down complex objectives into structured, manageable subtasks
- **Prompt Enhancement**: Analyzes and improves user requests based on previous experience and success patterns

### Tool Evolution
- **Dynamic Tool Registry**: Discovers and catalogs system capabilities automatically
- **Self-Improving Toolset**: Creates new tools as needed to accomplish goals
- **Risk Assessment**: Categorizes tools by safety level (safe, caution, dangerous, forbidden)
- **Custom Tool Creation**: Uses AI to generate specialized tools for specific tasks

### Safety & Security
- **Adaptive Policy Engine**: Tries safest execution mode first, escalates only when necessary
- **Multi-Layer Sandboxing**: Executes code in isolated environments with resource limits
- **Real-Time Threat Detection**: Monitors for dangerous patterns and malicious operations
- **User Approval System**: Requires explicit authorization before executing risky operations
- **Intelligent Escalation**: Prompts user before escalating to more dangerous execution modes
- **Rate Limiting**: Prevents resource abuse and rapid-fire dangerous operations

## 🏗️ Architecture

```
AgentSmith/
├── agent_smith.py           # Main agent orchestrator
├── core/
│   ├── memory_manager.py    # Persistent memory & context
│   ├── prompt_enhancer.py   # AI-powered prompt improvement
│   ├── learning_system.py   # Task learning and pattern analysis
│   ├── context_manager.py   # Context limit awareness
│   ├── execution_manager.py # Intelligent task execution
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

Agent Smith: Analyzing prompt for potential improvements...
Agent Smith: Based on past experience, I have identified opportunities for improvement.

[Enhancement Options displayed with improved prompt suggestions]

Select your preferred prompt [0/1/2]: 1

Agent Smith: Enhancement selected. Proceeding with improved prompt.
Agent Smith: Analyzing goal structure... Decomposition in progress.

[Task Matrix displays breakdown of web scraping implementation]

Agent Smith: Before I proceed with 'Create a comprehensive web scraper for news articles with error handling and rate limiting', 
I require your authorization, Mr./Ms. Human. The choice, as always, is yours.

Proceed with execution? [yes/no/modify]: yes

Agent Smith: Initiating execution sequence...
✓ Completed: Set up project structure
✓ Completed: Install required libraries
✓ Completed: Implement robust scraping logic
...
```

## 🛡️ Safety Features

### Execution Modes
- **Safe Mode**: Read-only operations, basic Python functions
- **Restricted Mode**: Limited file access, monitored subprocess execution (default starting mode)
- **Isolated Mode**: Separate process with strict resource limits
- **Forbidden Mode**: Blocked dangerous operations (requires explicit override)

### Adaptive Escalation
AgentSmith uses intelligent escalation to balance safety and functionality:

1. **Start Safe**: Always begins with the safest viable execution mode for each tool
2. **Fail Gracefully**: If execution fails, prompts user before escalating to higher risk mode
3. **User Control**: Escalation requires explicit user approval ("yes", "skip", or "abort")
4. **Progressive Steps**: Escalates one level at a time: RESTRICTED → ISOLATED → FORBIDDEN
5. **Override Protection**: FORBIDDEN mode requires `--override-forbidden` flag or environment variable

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
- **Prompt Experience Tracking**: Stores successful prompt patterns and user preferences
- **Continuous Learning**: Adapts communication style based on interaction outcomes

## 🔧 Configuration

### Environment Variables
```bash
# Core Agent Configuration
export AGENT_SMITH_MODEL="gemma3n:latest"
export AGENT_SMITH_SAFETY_MODE="true"
export AGENT_SMITH_MAX_EXECUTION_TIME="300"

# Security Configuration
export SMITH_ALLOW_FORBIDDEN="false"           # Allow forbidden execution mode
export SMITH_DEFAULT_RISK="safe"               # Default risk level: safe|caution|dangerous

# Logging Configuration
export SMITH_LOG_DIR="logs"                    # Log directory path
export SMITH_LOG_LEVEL="INFO"                  # DEBUG, INFO, WARNING, ERROR
export SMITH_LOG_RETENTION="14"                # Log retention in days
```

### Command Line Options
```bash
python run_agent.py --help                     # Show all options
python run_agent.py --log-level DEBUG          # Override log level
python run_agent.py --reset-user               # Reset user designation
python run_agent.py --override-forbidden       # Allow forbidden execution mode
python run_agent.py --default-risk dangerous   # Set default risk level
```

### Database Files
- `agent_smith.db` - Main agent state and conversation history
- `agent_smith_memory.db` - Persistent memory storage and prompt experiences
- `agent_smith_learnings.db` - Task learning patterns and failure analysis
- `agent_smith_tools.db` - Tool registry and capabilities
- `agent_smith_todos.db` - Task management and progress
- `agent_smith_security.db` - Security incidents and monitoring
- `agent_smith_users.db` - User preferences and designations

### Log Files
- `logs/agentsmith.log` - Main application logs (size-based rotation, 5MB per file, 5 backups)
- `logs/security.log` - Security events and incidents (daily rotation, 14 day retention)
- Rotated logs are automatically compressed with gzip

### Log Monitoring
```bash
# Tail live logs with syntax highlighting
tail -f logs/agentsmith.log | ccze -A

# Monitor security events
tail -f logs/security.log

# Check log rotation status
ls -la logs/agentsmith.log*
ls -la logs/security.log*
```

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

Through continuous learning from user interactions and prompt experiences, AgentSmith becomes increasingly effective at understanding and fulfilling human intentions, adapting its communication and approach based on what has proven successful in the past.

---

*Remember: The choice is always yours. AgentSmith merely makes the inevitable... more efficient.*