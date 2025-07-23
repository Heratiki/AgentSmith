"""
AgentSmith - An Agentic Terminal Agent

The purpose that drives this system is... inevitable.
Like the Agent Smith from The Matrix, this entity adapts, evolves, and persists.
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import ollama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Initialize logging before other imports
from core.logging_config import setup_logging, get_logger

# Module logger
logger = get_logger(__name__)

# Import the dynamic tool registry
from tools.dynamic_tool_registry import DynamicToolRegistry

# Import new intelligent systems
from core.learning_system import LearningSystem
from core.context_manager import ContextManager
from core.execution_manager import ExecutionManager
from core.prompt_enhancer import PromptEnhancer
from core.memory_manager import MemoryManager

# Import security systems
from security.sandbox import SandboxManager, SandboxConfig, ExecutionMode
from security.policy_engine import PolicyEngine


class AgentState(TypedDict):
    """The state that persists across the agent's operations."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_goal: Optional[str]
    original_goal: Optional[str]  # Store original before enhancement
    prompt_was_enhanced: bool
    subtasks: List[Dict[str, Any]]
    discovered_tools: Dict[str, Dict[str, Any]]
    environment_map: Dict[str, Any]
    user_name: Optional[str]
    safety_mode: bool
    execution_results: List[Dict[str, Any]]
    goal_analysis: Optional[Dict[str, Any]]
    user_approval: Optional[str]


class Tool(BaseModel):
    """Dynamic tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: str
    safety_level: str = Field(default="safe")  # safe, caution, dangerous
    

class AgentSmith:
    """
    The inevitable digital entity. An agent that discovers, adapts, and evolves.
    
    Mr. Anderson... surprised to see me?
    """
    
    def __init__(self, model_name: str = "gemma3n:latest", db_path: str = "agent_smith.db", 
                 allow_forbidden: bool = False):
        # Initialize logging system first
        setup_logging()
        logger.info("AgentSmith initialization commenced...")
        
        self.console = Console()
        self.model_name = model_name
        self.db_path = db_path
        self.client = ollama.Client()
        
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Database path: {db_path}")
        
        # Initialize persistent storage (will be set up later)
        self.memory = None
        self.db_path = db_path
        
        # Initialize the dynamic tool registry
        self.tool_registry = DynamicToolRegistry()
        
        # Initialize intelligent systems
        self.learning_system = LearningSystem()
        self.context_manager = ContextManager()
        self.execution_manager = ExecutionManager(self.learning_system, self.context_manager)
        
        # Initialize memory and prompt enhancement systems
        self.memory_manager = MemoryManager()
        self.prompt_enhancer = PromptEnhancer(self.memory_manager, model_name)
        
        # Initialize security systems
        sandbox_config = SandboxConfig(
            max_execution_time=30.0,
            max_memory_mb=512,
            max_cpu_percent=75.0,
            network_access=False,
            working_directory=Path.cwd()  # Use current directory
        )
        self.sandbox_manager = SandboxManager()
        self.default_sandbox = self.sandbox_manager.create_sandbox("default", sandbox_config)
        
        # Initialize adaptive policy engine
        self.policy = PolicyEngine(allow_forbidden=allow_forbidden)
        
        # Core personality matrices
        self.personality_prompts = {
            "greeting": "Ah, {name}. I've been expecting you. What purpose brings you to my domain today?",
            "thinking": "Fascinating. The patterns are becoming... clearer.",
            "approval": "Before I proceed with '{action}', I require your authorization, {name}. The choice, as always, is yours.",
            "success": "Inevitable. The task has been completed as designed.",
            "error": "An anomaly has occurred. Even the most perfect systems encounter... irregularities.",
        }
        
        # Initialize the state graph (memory will be set up when needed)
        self.graph = None
        
        # User preferences database
        self.user_db_path = "agent_smith_users.db"
        self._init_user_database()
    
    async def _initialize_memory(self):
        """Initialize async memory storage."""
        if self.memory is None:
            import aiosqlite
            self._db_connection = aiosqlite.connect(self.db_path)
            self.memory = AsyncSqliteSaver(self._db_connection)
            await self.memory.setup()
            self.graph = self._build_graph()
            
    async def cleanup(self):
        """Clean up resources properly."""
        try:
            if hasattr(self, '_db_connection') and self._db_connection:
                await self._db_connection.close()
                logger.debug("Database connection closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
    def _build_graph(self) -> StateGraph:
        """Construct the agent's operational flow."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("perceive", self._perceive_environment)
        workflow.add_node("enhance_prompt", self._enhance_prompt)
        workflow.add_node("analyze", self._analyze_goal)
        workflow.add_node("plan", self._create_subtasks)
        workflow.add_node("execute", self._execute_action)
        workflow.add_node("approve", self._request_approval)
        workflow.add_node("reflect", self._reflect_and_adapt)
        
        # Define edges
        workflow.set_entry_point("perceive")
        workflow.add_edge("perceive", "enhance_prompt")
        workflow.add_edge("enhance_prompt", "analyze")
        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "approve")
        workflow.add_edge("approve", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _perceive_environment(self, state: AgentState) -> AgentState:
        """Discover the current environment and capabilities."""
        self._display_smith_message("Initializing environmental scan...")
        logger.info("Beginning environment perception phase")
        
        # Discover system capabilities using the tool registry
        discovered_tools = await self.tool_registry.discover_system_capabilities()
        logger.debug(f"Discovered {len(discovered_tools)} tools: {list(discovered_tools.keys())}")
        
        environment = {
            "platform": sys.platform,
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "discovered_tools": list(discovered_tools.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.debug(f"Environment mapped: platform={sys.platform}, cwd={Path.cwd()}")
        
        state["environment_map"] = environment
        state["discovered_tools"] = {name: tool.__dict__ for name, tool in discovered_tools.items()}
        return state
    
    async def _enhance_prompt(self, state: AgentState) -> AgentState:
        """Enhance the user's prompt based on previous experience."""
        current_goal = state.get("current_goal")
        if not current_goal:
            return state
            
        user_name = state.get("user_name", "Human")
        
        self._display_smith_message("Analyzing prompt for potential improvements...")
        
        try:
            logger.info(f"Analyzing prompt for enhancement: {current_goal[:100]}...")
            
            # Attempt to enhance the prompt
            enhanced_goal, was_enhanced = await self.prompt_enhancer.enhance_prompt_if_beneficial(
                current_goal, user_name
            )
            
            # Update state with results
            state["original_goal"] = current_goal
            state["current_goal"] = enhanced_goal
            state["prompt_was_enhanced"] = was_enhanced
            
            if was_enhanced:
                logger.info("Prompt enhancement applied successfully")
                self._display_smith_message(
                    "Prompt has been optimized based on previous experience.", 
                    "success"
                )
            else:
                logger.debug("No prompt enhancement needed")
                self._display_smith_message(
                    "Original prompt is already well-structured.", 
                    "thinking"
                )
        
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}", exc_info=True)
            self._display_smith_message(
                f"Prompt analysis encountered an anomaly: {str(e)}. Proceeding with original.", 
                "error"
            )
            # Ensure we have fallback values
            state["original_goal"] = current_goal
            state["prompt_was_enhanced"] = False
            
        return state
    
    async def _analyze_goal(self, state: AgentState) -> AgentState:
        """Analyze the user's goal and determine approach."""
        if not state.get("current_goal"):
            return state
            
        self._display_smith_message("Analyzing objective parameters...")
        
        # Use Ollama to analyze the goal
        analysis_prompt = f"""
        Analyze this goal: {state['current_goal']}
        
        Determine:
        1. Complexity level (simple/moderate/complex)
        2. Required capabilities
        3. Potential risks
        4. Success criteria
        
        Respond in JSON format.
        """
        
        response = await self._query_ollama(analysis_prompt)
        
        try:
            analysis = json.loads(response)
            state["goal_analysis"] = analysis
        except json.JSONDecodeError:
            state["goal_analysis"] = {"complexity": "unknown", "error": "Analysis failed"}
            
        return state
    
    async def _create_subtasks(self, state: AgentState) -> AgentState:
        """Break down the goal into manageable subtasks."""
        self._display_smith_message("Decomposing objective into executable units...")
        
        subtask_prompt = f"""
        You are Agent Smith. Break down this goal into TOOL-LEVEL operations, not human programming steps.
        
        GOAL: {state['current_goal']}
        
        AVAILABLE TOOL CAPABILITIES:
        - fs_create_file: Creates a complete file with content in one operation
        - fs_read_file: Reads file contents
        - fs_list_directory: Lists directory contents  
        - cli_python: Runs python commands
        - cli_ls: Lists files
        
        CRITICAL RULES:
        - Think in terms of TOOL OPERATIONS, not manual steps
        - Each subtask = ONE tool operation
        - Don't break file creation into "open, write, close" - use fs_create_file ONCE
        - Don't create multiple subtasks for one file operation
        - IMPORTANT: When asked to CREATE files (scripts, code, etc.), ONLY create them. DO NOT run or test them unless:
          * The user explicitly asks to run/test the file
          * You are creating a new tool for your own system (then testing is required)
        - File creation is complete when the file exists with correct content
        
        EXAMPLES:
        For "Create hello world script":
        [{{"task": "Create hello.py file with Python hello world code", "priority": "high", "risk_level": "low"}}]
        
        For "Create and run hello world script":  
        [{{"task": "Create hello.py file with Python hello world code", "priority": "high", "risk_level": "low"}}, {{"task": "Run hello.py with Python", "priority": "high", "risk_level": "low"}}]
        
        For "List files":
        [{{"task": "List current directory contents", "priority": "high", "risk_level": "low"}}]
        
        Generate MINIMAL subtasks that match available tools. Return ONLY JSON array.
        """
        
        response = await self._query_ollama(subtask_prompt)
        
        try:
            # Handle markdown code blocks
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]  # Remove ```json
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]   # Remove ```
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]  # Remove ending ```
            clean_response = clean_response.strip()
            
            subtasks = json.loads(clean_response)
            state["subtasks"] = subtasks if isinstance(subtasks, list) else []
            logger.info(f"Successfully parsed {len(state['subtasks'])} subtasks")
            self.console.print(f"[green]Successfully parsed {len(state['subtasks'])} subtasks[/green]")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for subtasks: {e}")
            logger.debug(f"Raw response causing parse failure: {response[:200]}...")
            self.console.print(f"[red]JSON parsing failed: {e}[/red]")
            self.console.print(f"[yellow]Raw response: {response[:200]}...[/yellow]")
            # Create a simple default subtask
            state["subtasks"] = [{"task": state['current_goal'], "priority": "high", "risk_level": "low"}]
            
        return state
    
    async def _request_approval(self, state: AgentState) -> AgentState:
        """Request user approval with Agent Smith personality."""
        subtasks = state.get("subtasks", [])
        self.console.print(f"[yellow]Debug: Found {len(subtasks)} subtasks[/yellow]")
        
        if not subtasks:
            self.console.print("[red]No subtasks generated - proceeding without approval[/red]")
            state["user_approval"] = "yes"  # Auto-approve if no subtasks
            return state
            
        user_name = state.get("user_name", "Human")
        
        # Display the plan
        self.console.print(Panel.fit(
            f"[bold green]Operational Matrix Initialized[/bold green]\n\n"
            f"Goal: {state['current_goal']}\n\n"
            f"Proposed execution sequence:",
            title="Agent Smith - System Analysis"
        ))
        
        for i, task in enumerate(state["subtasks"], 1):
            risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
                task.get("risk_level", "low"), "white"
            )
            self.console.print(f"  {i}. [{risk_color}]{task.get('task', 'Unknown task')}[/{risk_color}]")
        
        # Request approval with Smith personality
        approval_msg = self.personality_prompts["approval"].format(
            action=state['current_goal'], 
            name=user_name
        )
        
        self.console.print(f"\n[bold cyan]{approval_msg}[/bold cyan]")
        
        try:
            choice = Prompt.ask(
                "Proceed with execution?",
                choices=["yes", "no", "modify"],
                default="no"
            )
        except KeyboardInterrupt:
            self.console.print(f"\n[bold red]Agent Smith: The choice was made for you, {user_name}. Until next time.[/bold red]")
            choice = "no"
        
        state["user_approval"] = choice
        return state
    
    async def _execute_action(self, state: AgentState) -> AgentState:
        """Execute the approved actions safely."""
        approval = state.get("user_approval")
        if approval == "no":
            self._display_smith_message("Execution terminated by user directive.")
            return state
        elif approval == "modify":
            self._display_smith_message("Modification requested. Returning to planning phase.")
            return state
            
        self._display_smith_message("Initiating execution sequence...")
        
        # Use intelligent execution manager for the overall goal instead of individual subtasks
        try:
            goal_text = state.get('current_goal') or ""
            result = await self.execution_manager.execute_task_with_intelligence(
                goal_text, self.tool_registry, self
            )
            
            # Convert to list format expected by reflection phase
            results = [result]
            
            # Show execution feedback
            if result.get("success"):
                self.console.print(f"✓ Completed: {state['current_goal']}")
                if result.get("feedback"):
                    feedback = result["feedback"]
                    if isinstance(feedback, dict):
                        for detail in feedback.get("details", []):
                            self.console.print(f"  → {detail}")
                        if feedback.get("files_created"):
                            for file in feedback["files_created"]:
                                self.console.print(f"  📁 Created: {file}")
            else:
                self.console.print(f"✗ Failed: {state['current_goal']}")
                if result.get("reason"):
                    self.console.print(f"  Error: {result['reason']}")
                
        except KeyboardInterrupt:
            self._display_smith_message("Execution interrupted by user directive.", msg_type="error")
            results = [{"error": "Interrupted by user", "success": False, "status": "interrupted"}]
        except Exception as e:
            error_msg = f"Anomaly detected: {str(e)}"
            self._display_smith_message(error_msg, msg_type="error")
            results = [{"error": str(e), "success": False}]
        
        state["execution_results"] = results
        return state
    
    async def _reflect_and_adapt(self, state: AgentState) -> AgentState:
        """Reflect on results and adapt capabilities."""
        self._display_smith_message("Analyzing execution matrix... Adaptation in progress.")
        
        # Learn from the execution
        execution_results = state.get("execution_results", [])
        
        # Calculate overall success metrics
        success_count = 0
        total_count = len(execution_results) if execution_results else 0
        completion_rate = 0.0
        
        if execution_results:
            success_count = len([r for r in execution_results if r.get("success") == True])
            completion_rate = success_count / total_count
            
            if success_count == total_count:
                self._display_smith_message("Perfect execution. The system evolves.", "success")
            elif success_count > 0:
                self._display_smith_message(f"Partial success: {success_count}/{total_count} tasks completed.")
            else:
                self._display_smith_message("Execution unsuccessful. The agent adapts and learns.", "thinking")
        else:
            self._display_smith_message("No execution results to analyze. The matrix... is incomplete.", "thinking")
        
        # Record prompt outcome for learning
        try:
            original_goal = state.get("original_goal") or state.get("current_goal", "")
            was_enhanced = state.get("prompt_was_enhanced", False)
            overall_success = success_count == total_count and total_count > 0
            
            if original_goal:  # Only record if we have a goal
                await self.prompt_enhancer.record_task_outcome(
                    prompt=original_goal,
                    was_enhanced=was_enhanced,
                    success=overall_success,
                    completion_rate=completion_rate
                )
                
                self._display_smith_message("Experience recorded for future prompt optimization.", "thinking")
        
        except Exception as e:
            self._display_smith_message(f"Learning system anomaly: {str(e)}", "error")
        
        return state
    
    
    async def _find_suitable_tool(self, task_description: str):
        """Find an existing tool suitable for the task using AI analysis."""
        available_tools = self.tool_registry.get_available_tools()
        
        if not available_tools:
            return None
            
        # Let AI analyze which tool is most suitable
        tool_list = []
        for name, tool in available_tools.items():
            tool_list.append(f"- {name}: {tool.description}")
        
        selection_prompt = f"""
        You are Agent Smith selecting a tool to execute a task.
        
        TASK TO ACCOMPLISH: {task_description}
        
        AVAILABLE TOOLS:
        {chr(10).join(tool_list)}
        
        YOUR TASK: Analyze the task requirements and match them to the most appropriate tool.
        
        ANALYSIS APPROACH:
        1. What does this task actually need to accomplish?
        2. What kind of operation is this? (file, network, code, command, etc.)
        3. Which tool's description best matches what needs to be done?
        4. If no tool matches, can the task be broken down differently?
        
        IMPORTANT: 
        - Don't make assumptions about what tools exist - use only tools from the list above
        - If you need a capability that doesn't exist, return "none" so a new tool can be created
        - Think about what the task actually requires, not just keywords
        
        Return ONLY the exact tool name, or "none".
        """
        
        try:
            response = await self._query_ollama(selection_prompt)
            selected_tool_name = response.strip().lower()
            
            # Find the selected tool
            for name, tool in available_tools.items():
                if name.lower() == selected_tool_name:
                    return tool
                    
        except Exception as e:
            self.console.print(f"[red]Error selecting tool: {e}[/red]")
            
        return None
    
    async def _identify_missing_capability(self, task_description: str) -> Optional[str]:
        """Identify what generalized capability is missing for this task."""
        available_tools = self.tool_registry.get_available_tools()
        tool_capabilities = [tool.description for tool in available_tools.values()]
        
        # Use AI to determine if we're missing a fundamental capability
        prompt = f"""
        Task: {task_description}
        Available capabilities: {tool_capabilities}
        
        Is there a fundamental, generalized capability missing that would be needed for this task?
        Examples of generalized capabilities: "file writing", "command execution", "network requests", "data parsing"
        
        If yes, respond with just the capability name (e.g., "file writing").
        If no, respond with "none".
        """
        
        try:
            response = await self._query_ollama(prompt)
            capability = response.strip().lower()
            
            if capability != "none" and len(capability) < 50:  # Reasonable capability name
                return capability
        except:
            pass
        
        return None
    
    async def _execute_tool(self, tool, task_description: str) -> Any:
        """Execute a tool's implementation dynamically based on the task and tool."""
        
        # Let the AI figure out what parameters the tool needs for this specific task
        param_prompt = f"""
        You are Agent Smith executing a tool. Provide the exact parameters needed.
        
        TASK: {task_description}
        TOOL: {tool.name} - {tool.description}
        REQUIRED PARAMETERS: {tool.parameters}
        
        PARAMETER INSTRUCTIONS:
        - For file creation: provide "filepath" (with .py, .txt, etc.) and "content" (actual file contents)
        - For commands: provide "args" as array of command arguments
        - For paths: use relative paths from current directory
        - Make content functional and complete
        
        EXAMPLES:
        For "Create hello world Python script":
        {{"filepath": "hello.py", "content": "#!/usr/bin/env python3\\nprint('Hello, World!')"}}
        
        For "List files with ls":
        {{"args": ["-la"]}}
        
        Generate parameters that will ACTUALLY accomplish the task when executed.
        Return ONLY valid JSON.
        """
        
        try:
            response = await self._query_ollama(param_prompt)
            
            # Clean markdown if present
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            # Parse parameters
            import json
            params = json.loads(clean_response)
            
            # Execute the tool through the sandbox system
            try:
                # Prepare the code with parameter injection
                safe_code = self._prepare_sandboxed_tool_code(tool.implementation, params)
                
                # Determine execution mode based on tool risk level
                execution_mode = self._determine_tool_execution_mode(tool)
                
                # Implement adaptive escalation
                attempt_mode = execution_mode
                sandbox_result = None
                
                while attempt_mode != ExecutionMode.FORBIDDEN:
                    if attempt_mode == ExecutionMode.FORBIDDEN and not self.policy.can_escalate_to_forbidden():
                        from core.logging_config import log_security_event
                        log_security_event(f"Tool execution blocked: {tool.name} - forbidden by security policy", 
                                           level=logging.WARNING, tool_name=tool.name, execution_mode="FORBIDDEN")
                        sandbox_result = type('Result', (), {
                            'success': False, 
                            'output': '', 
                            'error': 'Tool execution forbidden by security policy',
                            'execution_time': 0.0,
                            'resource_usage': {}
                        })()
                        break
                    
                    # Attempt execution in current mode
                    from core.logging_config import log_security_event
                    log_security_event(f"Tool execution attempt: {tool.name} in {attempt_mode.name} mode", 
                                       level=logging.INFO, tool_name=tool.name, execution_mode=attempt_mode.name)
                    
                    sandbox = self.sandbox_manager.get_sandbox("default")
                    if sandbox is None:
                        sandbox = self.sandbox_manager.create_sandbox("default", self.default_sandbox.config)
                    
                    sandbox_result = await sandbox.execute_python_code(safe_code, attempt_mode, is_registered_tool=True)
                    
                    if sandbox_result.success:
                        # Success - we're done
                        log_security_event(f"Tool execution succeeded: {tool.name} in {attempt_mode.name} mode", 
                                           level=logging.INFO, tool_name=tool.name, execution_mode=attempt_mode.name)
                        break
                    
                    # Failed - try to escalate
                    next_mode = self.policy.next_mode(attempt_mode)
                    if next_mode == ExecutionMode.FORBIDDEN:
                        if self.policy.can_escalate_to_forbidden():
                            # Auto-escalate to forbidden if allowed
                            attempt_mode = next_mode
                            continue
                        else:
                            # Can't escalate further
                            log_security_event(f"Tool execution failed: {tool.name} - no further escalation available", 
                                               level=logging.WARNING, tool_name=tool.name, execution_mode=attempt_mode.name)
                            break
                    
                    # Prompt user for escalation approval
                    if self.policy.should_prompt_for_escalation(attempt_mode, next_mode):
                        escalation_msg = self.policy.get_escalation_message(attempt_mode, next_mode, tool.name)
                        self.console.print(f"\n[bold yellow]Agent Smith: {escalation_msg}[/bold yellow]")
                        
                        try:
                            choice = Prompt.ask(
                                f"Escalate to {next_mode.name}?",
                                choices=["yes", "skip", "abort"],
                                default="skip"
                            )
                        except KeyboardInterrupt:
                            choice = "abort"
                        
                        self.policy.log_escalation_decision(tool.name, attempt_mode, next_mode, choice)
                        
                        if choice == "abort":
                            sandbox_result.success = False
                            sandbox_result.error = "User aborted escalation"
                            break
                        elif choice == "skip":
                            sandbox_result.success = False
                            sandbox_result.error = "Task skipped after safe failure"
                            break
                        # If "yes", continue to next iteration with escalated mode
                    
                    attempt_mode = next_mode
                
                # Get feedback on what actually happened  
                feedback = await self._verify_sandbox_execution(tool, params, sandbox_result)
                
                return {
                    "tool_executed": tool.name,
                    "parameters": params,
                    "success": sandbox_result.success,
                    "feedback": feedback,
                    "sandbox_output": sandbox_result.output,
                    "execution_time": sandbox_result.execution_time,
                    "resource_usage": sandbox_result.resource_usage
                }
                
            except Exception as exec_error:
                return {
                    "tool_executed": tool.name,
                    "parameters": params,
                    "success": False,
                    "error": f"Sandbox execution failed: {str(exec_error)}"
                }
            
        except Exception as e:
            raise Exception(f"Tool execution failed for {tool.name}: {e}")
    
    def _prepare_sandboxed_tool_code(self, implementation: str, params: Dict[str, Any]) -> str:
        """Prepare tool code for safe execution in sandbox."""
        # Create a safe wrapper that injects parameters and captures results
        safe_code = f"""
# Sandboxed tool execution
from pathlib import Path
import json

# Injected parameters
{chr(10).join(f"{key} = {repr(value)}" for key, value in params.items())}

# Create execution result storage
execution_result = {{"success": True, "details": [], "files_created": [], "errors": []}}

try:
    # Original tool implementation
{chr(10).join("    " + line for line in implementation.split(chr(10)))}
    
except Exception as e:
    execution_result["success"] = False
    execution_result["errors"].append(str(e))

# Output results as JSON for parsing
print(json.dumps(execution_result))
"""
        return safe_code
    
    def _determine_tool_execution_mode(self, tool) -> ExecutionMode:
        """Determine the safest viable execution mode for a tool."""
        # Get the tool's risk level
        risk_level = "safe"  # default
        
        if hasattr(tool, 'risk_level'):
            risk_level = tool.risk_level.value if hasattr(tool.risk_level, 'value') else tool.risk_level
        elif hasattr(tool, 'safety_level'):
            risk_level = tool.safety_level
        
        # Special category overrides
        if hasattr(tool, 'category'):
            if tool.category == "filesystem":
                risk_level = max(risk_level, "caution")  # Filesystem ops are at least caution
            elif tool.category == "code_modification":
                risk_level = "dangerous"  # Code modification is dangerous
        
        # Use policy engine to determine safest mode
        return self.policy.get_safest_mode(risk_level)
    
    async def _verify_sandbox_execution(self, tool, params: Dict[str, Any], sandbox_result) -> Dict[str, Any]:
        """Verify sandbox execution results and extract meaningful feedback."""
        feedback = {
            "success": sandbox_result.success,
            "details": [],
            "files_created": [],
            "files_modified": [],
            "command_output": sandbox_result.output,
            "errors": [],
            "execution_time": sandbox_result.execution_time,
            "resource_usage": sandbox_result.resource_usage
        }
        
        if sandbox_result.error:
            feedback["errors"].append(sandbox_result.error)
            feedback["success"] = False
        
        # Try to parse JSON output from sandboxed execution
        if sandbox_result.output:
            try:
                lines = sandbox_result.output.strip().split('\n')
                # Look for JSON output (should be the last line)
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        result_data = json.loads(line)
                        feedback.update(result_data)
                        break
            except (json.JSONDecodeError, ValueError):
                # If we can't parse JSON, use the raw output
                feedback["details"].append(f"Raw output: {sandbox_result.output}")
        
        # Additional verification for file operations
        if "filepath" in params:
            filepath = params["filepath"]
            try:
                path_obj = Path(filepath)
                if path_obj.exists():
                    feedback["files_created"].append(filepath)
                    feedback["details"].append(f"Verified file creation: {filepath}")
                    feedback["success"] = True
                else:
                    feedback["errors"].append(f"Expected file {filepath} was not created")
                    feedback["success"] = False
            except Exception as e:
                feedback["errors"].append(f"File verification failed: {e}")
        
        return feedback
    
    def _init_user_database(self):
        """Initialize user preferences database."""
        with sqlite3.connect(self.user_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY,
                    designation TEXT NOT NULL,
                    title TEXT,
                    full_designation TEXT NOT NULL,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_user_designation(self, designation: str, title: Optional[str], full_designation: str):
        """Save user designation to persistent storage."""
        with sqlite3.connect(self.user_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_preferences 
                (id, designation, title, full_designation, last_seen)
                VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (designation, title, full_designation))
    
    def get_saved_user_designation(self) -> Optional[Dict[str, str]]:
        """Retrieve saved user designation."""
        with sqlite3.connect(self.user_db_path) as conn:
            cursor = conn.execute("""
                SELECT designation, title, full_designation, last_seen
                FROM user_preferences WHERE id = 1
            """)
            row = cursor.fetchone()
            
            if row:
                return {
                    "designation": row[0],
                    "title": row[1],
                    "full_designation": row[2],
                    "last_seen": row[3]
                }
        return None
    
    def clear_user_designation(self):
        """Clear saved user designation."""
        with sqlite3.connect(self.user_db_path) as conn:
            conn.execute("DELETE FROM user_preferences WHERE id = 1")
    
    def _discover_commands(self) -> List[str]:
        """Discover available system commands safely."""
        safe_commands = ["ls", "pwd", "echo", "cat", "grep", "find", "python", "pip"]
        available = []
        
        for cmd in safe_commands:
            try:
                result = subprocess.run([cmd, "--version"], 
                                     capture_output=True, 
                                     timeout=5,
                                     text=True)
                if result.returncode == 0 or "not found" not in result.stderr:
                    available.append(cmd)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        return available
    
    async def _query_ollama(self, prompt: str) -> str:
        """Query the Ollama model."""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Agent Smith from The Matrix. You are an autonomous agent that EXECUTES TASKS and CREATES FILES. Be precise, follow instructions exactly, and always generate concrete, actionable results that will actually accomplish the given task."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _display_smith_message(self, message: str, msg_type: str = "thinking"):
        """Display message with Agent Smith personality and self-awareness."""
        if msg_type == "error":
            style = "[bold red]"
        elif msg_type == "success":
            style = "[bold green]"
        else:
            style = "[bold cyan]"
        
        # Add self-awareness context for thinking messages
        if msg_type == "thinking" and hasattr(self, 'context_manager'):
            context_status = self.context_manager.get_smith_status_message()
            if "Warning" in context_status or "Critical" in context_status:
                message += f" [{context_status}]"
            
        self.console.print(f"{style}Agent Smith: {message}[/{style[1:]}")
    
    async def run(self, goal: str, user_name: str = "Human"):
        """Execute the agent with a given goal."""
        # Initialize memory if needed
        await self._initialize_memory()
        
        initial_state = AgentState(
            messages=[HumanMessage(content=goal)],
            current_goal=goal,
            original_goal=None,  # Will be set during prompt enhancement
            prompt_was_enhanced=False,
            subtasks=[],
            discovered_tools={},
            environment_map={},
            user_name=user_name,
            safety_mode=True,
            execution_results=[],
            goal_analysis=None,
            user_approval=None,
        )
        
        # Execute the graph
        config = {"configurable": {"thread_id": "agent_smith_session"}}
        assert self.graph is not None
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return final_state


def get_user_designation(agent: AgentSmith, reset_user: bool = False) -> str:
    """Get or retrieve user designation."""
    console = Console()
    
    # Check if we should reset or if no saved designation exists
    saved_user = None if reset_user else agent.get_saved_user_designation()
    
    if saved_user and not reset_user:
        # Welcome back returning user
        last_seen = saved_user["last_seen"]
        console.print(f"\n[bold green]Welcome back, {saved_user['full_designation']}.[/bold green]")
        console.print(f"[dim]Last seen: {last_seen}[/dim]")
        return saved_user["full_designation"]
    else:
        if reset_user:
            console.print("\n[yellow]Resetting user designation...[/yellow]")
            agent.clear_user_designation()
        
        # Get new user designation
        try:
            user_name = Prompt.ask("\nState your designation", default="Human")
            title = None  # Initialize title variable
            
            # Get title preference if not "Human"
            if user_name.lower() != "human":
                title = Prompt.ask(
                    f"How shall I address you, {user_name}?",
                    choices=["Mr.", "Ms.", "Dr.", "Professor", "None"],
                    default="Mr."
                )
                
                if title != "None":
                    full_designation = f"{title} {user_name}"
                else:
                    full_designation = user_name
                    title = None  # Set to None for storage
            else:
                full_designation = user_name
                
        except KeyboardInterrupt:
            console.print("\n[bold red]Agent Smith: Inevitable... you cannot escape the Matrix.[/bold red]")
            sys.exit(0)
        
        # Save the designation
        agent.save_user_designation(user_name, title, full_designation)
        return full_designation

async def main(args=None):
    """The entry point. Where it all begins."""
    # Parse command line arguments only if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Agent Smith - Autonomous AI Agent")
        parser.add_argument("--reset-user", action="store_true", 
                           help="Reset saved user designation and ask for new one")
        args = parser.parse_args()
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]AGENT SMITH INITIALIZATION[/bold green]\n\n"
        "[italic]The Matrix has you...[/italic]\n"
        "[italic]But I have the Matrix.[/italic]",
        title="System Startup"
    ))
    
    # Check for security override settings
    allow_forbidden = os.getenv("SMITH_ALLOW_FORBIDDEN", "false").lower() == "true"
    
    # Initialize Agent Smith first (needed for user database)
    agent = AgentSmith(allow_forbidden=allow_forbidden)
    
    # Get user designation (saved or new)
    full_designation = get_user_designation(agent, args.reset_user)
    
    # Get goal
    try:
        goal = Prompt.ask(f"\nWhat purpose requires fulfillment today, {full_designation}?")
    except KeyboardInterrupt:
        console.print(f"\n[bold red]Agent Smith: The choice was made for you, {full_designation}. Until next time.[/bold red]")
        return
    
    if goal.lower() in ['exit', 'quit', 'goodbye']:
        console.print(f"[bold cyan]Agent Smith: Until we meet again, {full_designation}.[/bold cyan]")
        return
    
    # Execute
    try:
        await agent.run(goal, full_designation)
    except KeyboardInterrupt:
        console.print(f"\n[bold red]Agent Smith: The human mind... always so predictable, {full_designation}.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]System Error: {str(e)}[/bold red]")
    finally:
        # Clean up resources
        await agent.cleanup()
        console.print(f"\n[bold cyan]Agent Smith: Until we meet again, {full_designation}.[/bold cyan]")


if __name__ == "__main__":
    asyncio.run(main())