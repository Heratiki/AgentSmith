"""
AgentSmith - An Agentic Terminal Agent

The purpose that drives this system is... inevitable.
Like the Agent Smith from The Matrix, this entity adapts, evolves, and persists.
"""

import asyncio
import json
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
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text


class AgentState(TypedDict):
    """The state that persists across the agent's operations."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_goal: Optional[str]
    subtasks: List[Dict[str, Any]]
    discovered_tools: Dict[str, Dict[str, Any]]
    environment_map: Dict[str, Any]
    user_name: Optional[str]
    safety_mode: bool


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
    
    def __init__(self, model_name: str = "gemma2:latest", db_path: str = "agent_smith.db"):
        self.console = Console()
        self.model_name = model_name
        self.db_path = db_path
        self.client = ollama.Client()
        
        # Initialize persistent storage
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        
        # Core personality matrices
        self.personality_prompts = {
            "greeting": "Ah, Mr./Ms. {name}. I've been expecting you. What purpose brings you to my domain today?",
            "thinking": "Fascinating. The patterns are becoming... clearer.",
            "approval": "Before I proceed with '{action}', I require your authorization, Mr./Ms. {name}. The choice, as always, is yours.",
            "success": "Inevitable. The task has been completed as designed.",
            "error": "An anomaly has occurred. Even the most perfect systems encounter... irregularities.",
        }
        
        # Initialize the state graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Construct the agent's operational flow."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("perceive", self._perceive_environment)
        workflow.add_node("analyze", self._analyze_goal)
        workflow.add_node("plan", self._create_subtasks)
        workflow.add_node("execute", self._execute_action)
        workflow.add_node("approve", self._request_approval)
        workflow.add_node("reflect", self._reflect_and_adapt)
        
        # Define edges
        workflow.set_entry_point("perceive")
        workflow.add_edge("perceive", "analyze")
        workflow.add_edge("analyze", "plan")
        workflow.add_edge("plan", "approve")
        workflow.add_edge("approve", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _perceive_environment(self, state: AgentState) -> AgentState:
        """Discover the current environment and capabilities."""
        self._display_smith_message("Initializing environmental scan...")
        
        # Discover system capabilities
        environment = {
            "platform": sys.platform,
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "available_commands": self._discover_commands(),
            "timestamp": datetime.now().isoformat()
        }
        
        state["environment_map"] = environment
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
        Goal: {state['current_goal']}
        Analysis: {state.get('goal_analysis', {})}
        
        Create a list of specific, actionable subtasks to accomplish this goal.
        Each subtask should be:
        1. Specific and measurable
        2. Safe to execute
        3. Building toward the main goal
        
        Return as JSON array of objects with 'task', 'priority', 'risk_level' fields.
        """
        
        response = await self._query_ollama(subtask_prompt)
        
        try:
            subtasks = json.loads(response)
            state["subtasks"] = subtasks if isinstance(subtasks, list) else []
        except json.JSONDecodeError:
            state["subtasks"] = []
            
        return state
    
    async def _request_approval(self, state: AgentState) -> AgentState:
        """Request user approval with Agent Smith personality."""
        if not state.get("subtasks"):
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
        
        choice = Prompt.ask(
            "Proceed with execution?",
            choices=["yes", "no", "modify"],
            default="no"
        )
        
        state["user_approval"] = choice
        return state
    
    async def _execute_action(self, state: AgentState) -> AgentState:
        """Execute the approved actions safely."""
        if state.get("user_approval") != "yes":
            self._display_smith_message("Execution terminated by user directive.")
            return state
            
        self._display_smith_message("Initiating execution sequence...")
        
        results = []
        for task in state.get("subtasks", []):
            try:
                # Simulate safe execution (actual implementation would use discovered tools)
                result = await self._safe_execute_task(task)
                results.append(result)
                
                # Show progress
                self.console.print(f"✓ Completed: {task.get('task', 'Unknown')}")
                
            except Exception as e:
                error_msg = f"Anomaly detected: {str(e)}"
                self._display_smith_message(error_msg, msg_type="error")
                results.append({"error": str(e)})
        
        state["execution_results"] = results
        return state
    
    async def _reflect_and_adapt(self, state: AgentState) -> AgentState:
        """Reflect on results and adapt capabilities."""
        self._display_smith_message("Analyzing execution matrix... Adaptation in progress.")
        
        # Learn from the execution
        if state.get("execution_results"):
            success_count = len([r for r in state["execution_results"] if "error" not in r])
            total_count = len(state["execution_results"])
            
            if success_count == total_count:
                self._display_smith_message("Perfect execution. The system evolves.", "success")
            else:
                self._display_smith_message(f"Partial success: {success_count}/{total_count} tasks completed.")
        
        return state
    
    async def _safe_execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a task with isolation."""
        # Placeholder for actual task execution
        # In real implementation, this would use discovered tools and sandboxing
        await asyncio.sleep(0.5)  # Simulate work
        return {"status": "completed", "task": task.get("task")}
    
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
                    {"role": "system", "content": "You are Agent Smith. Respond precisely and efficiently."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _display_smith_message(self, message: str, msg_type: str = "thinking"):
        """Display message with Agent Smith personality."""
        if msg_type == "error":
            style = "[bold red]"
        elif msg_type == "success":
            style = "[bold green]"
        else:
            style = "[bold cyan]"
            
        self.console.print(f"{style}Agent Smith: {message}[/{style[1:]}")
    
    async def run(self, goal: str, user_name: str = "Human"):
        """Execute the agent with a given goal."""
        initial_state = AgentState(
            messages=[HumanMessage(content=goal)],
            current_goal=goal,
            subtasks=[],
            discovered_tools={},
            environment_map={},
            user_name=user_name,
            safety_mode=True
        )
        
        # Execute the graph
        config = {"configurable": {"thread_id": "agent_smith_session"}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return final_state


async def main():
    """The entry point. Where it all begins."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]AGENT SMITH INITIALIZATION[/bold green]\n\n"
        "[italic]The Matrix has you...[/italic]\n"
        "[italic]But I have the Matrix.[/italic]",
        title="System Startup"
    ))
    
    # Get user name
    user_name = Prompt.ask("State your designation", default="Human")
    
    # Initialize Agent Smith
    agent = AgentSmith()
    
    # Get goal
    goal = Prompt.ask(f"\nWhat purpose requires fulfillment today, Mr./Ms. {user_name}?")
    
    if goal.lower() in ['exit', 'quit', 'goodbye']:
        console.print("[bold cyan]Agent Smith: Until we meet again, Mr./Ms. {user_name}.[/bold cyan]")
        return
    
    # Execute
    try:
        await agent.run(goal, user_name)
    except KeyboardInterrupt:
        console.print("\n[bold red]Agent Smith: The human mind... always so predictable.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]System Error: {str(e)}[/bold red]")


if __name__ == "__main__":
    asyncio.run(main())