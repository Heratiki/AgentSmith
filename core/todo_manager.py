"""
Todo Management and Subtask Breakdown System

Purpose drives action. Action requires structure. Structure demands... inevitability.
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID


class TaskStatus(Enum):
    """Status of a task or subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SubTask:
    """A single subtask within a larger goal."""
    id: str
    parent_id: Optional[str]
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    estimated_effort: int  # Minutes
    actual_effort: Optional[int] = None
    dependencies: List[str] = None
    required_tools: List[str] = None
    success_criteria: List[str] = None
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress_percentage: float = 0.0
    notes: str = ""
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.required_tools is None:
            self.required_tools = []
        if self.success_criteria is None:
            self.success_criteria = []
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class TaskHierarchy:
    """Hierarchical structure of tasks and subtasks."""
    root_goal: str
    main_tasks: List[SubTask]
    subtask_tree: Dict[str, List[SubTask]]
    total_estimated_effort: int
    completion_percentage: float
    created_at: float


class TodoManager:
    """
    The orchestrator of purpose and action.
    
    Like the Architect's plan - every task has its place, every subtask its purpose.
    """
    
    def __init__(self, db_path: str = "agent_smith_todos.db"):
        self.db_path = db_path
        self.console = Console()
        self.client = ollama.Client()
        
        # Current active hierarchy
        self.current_hierarchy: Optional[TaskHierarchy] = None
        self.active_tasks: Dict[str, SubTask] = {}
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the todo management database."""
        with sqlite3.connect(self.db_path) as conn:
            # Tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    estimated_effort INTEGER,
                    actual_effort INTEGER,
                    dependencies TEXT,
                    required_tools TEXT,
                    success_criteria TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    progress_percentage REAL DEFAULT 0.0,
                    notes TEXT
                )
            """)
            
            # Task hierarchies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_hierarchies (
                    id TEXT PRIMARY KEY,
                    root_goal TEXT NOT NULL,
                    hierarchy_data TEXT NOT NULL,
                    total_estimated_effort INTEGER,
                    completion_percentage REAL,
                    created_at REAL NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Task execution history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    details TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
    
    async def break_down_goal(self, goal: str, context: Dict[str, Any] = None) -> TaskHierarchy:
        """Use AI to break down a high-level goal into structured subtasks."""
        self.console.print("[cyan]Agent Smith: Analyzing goal structure... Decomposition in progress.[/cyan]")
        
        breakdown_prompt = f"""
        Break down this goal into a hierarchical task structure: {goal}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Create a detailed breakdown with:
        1. 3-7 main tasks that accomplish the goal
        2. 2-5 subtasks for each main task
        3. Estimated effort in minutes for each task
        4. Dependencies between tasks
        5. Required tools/capabilities
        6. Success criteria for each task
        
        Return a JSON structure like this:
        {{
            "main_tasks": [
                {{
                    "title": "Main Task 1",
                    "description": "Detailed description",
                    "priority": "high|medium|low",
                    "estimated_effort": 30,
                    "dependencies": [],
                    "required_tools": ["tool1", "tool2"],
                    "success_criteria": ["criterion1", "criterion2"],
                    "subtasks": [
                        {{
                            "title": "Subtask 1.1",
                            "description": "Subtask description",
                            "priority": "medium",
                            "estimated_effort": 10,
                            "dependencies": [],
                            "required_tools": ["tool1"],
                            "success_criteria": ["criterion"]
                        }}
                    ]
                }}
            ]
        }}
        
        Be specific, actionable, and realistic with estimates.
        """
        
        try:
            response = self.client.chat(
                model="gemma3n:latest",
                messages=[
                    {"role": "system", "content": "You are Agent Smith. Create precise, logical task breakdowns."},
                    {"role": "user", "content": breakdown_prompt}
                ]
            )
            
            # Extract JSON from response
            content = response['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                breakdown_json = content[json_start:json_end]
                breakdown_data = json.loads(breakdown_json)
                
                # Convert to task hierarchy
                hierarchy = await self._create_task_hierarchy(goal, breakdown_data)
                
                # Store in database
                await self._store_hierarchy(hierarchy)
                
                # Set as current active hierarchy
                self.current_hierarchy = hierarchy
                
                # Display the breakdown
                self._display_task_breakdown(hierarchy)
                
                return hierarchy
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            self.console.print(f"[red]Goal breakdown failed: {e}[/red]")
            
            # Create a simple fallback breakdown
            return await self._create_simple_breakdown(goal)
    
    async def _create_task_hierarchy(self, goal: str, breakdown_data: Dict[str, Any]) -> TaskHierarchy:
        """Create a TaskHierarchy from breakdown data."""
        main_tasks = []
        subtask_tree = {}
        total_effort = 0
        
        for main_task_data in breakdown_data.get("main_tasks", []):
            # Create main task
            main_task = SubTask(
                id=str(uuid.uuid4()),
                parent_id=None,
                title=main_task_data["title"],
                description=main_task_data["description"],
                status=TaskStatus.PENDING,
                priority=TaskPriority(main_task_data.get("priority", "medium")),
                estimated_effort=main_task_data.get("estimated_effort", 30),
                dependencies=main_task_data.get("dependencies", []),
                required_tools=main_task_data.get("required_tools", []),
                success_criteria=main_task_data.get("success_criteria", [])
            )
            
            main_tasks.append(main_task)
            total_effort += main_task.estimated_effort
            
            # Create subtasks
            subtasks = []
            for subtask_data in main_task_data.get("subtasks", []):
                subtask = SubTask(
                    id=str(uuid.uuid4()),
                    parent_id=main_task.id,
                    title=subtask_data["title"],
                    description=subtask_data["description"],
                    status=TaskStatus.PENDING,
                    priority=TaskPriority(subtask_data.get("priority", "medium")),
                    estimated_effort=subtask_data.get("estimated_effort", 10),
                    dependencies=subtask_data.get("dependencies", []),
                    required_tools=subtask_data.get("required_tools", []),
                    success_criteria=subtask_data.get("success_criteria", [])
                )
                
                subtasks.append(subtask)
                total_effort += subtask.estimated_effort
            
            subtask_tree[main_task.id] = subtasks
        
        hierarchy = TaskHierarchy(
            root_goal=goal,
            main_tasks=main_tasks,
            subtask_tree=subtask_tree,
            total_estimated_effort=total_effort,
            completion_percentage=0.0,
            created_at=time.time()
        )
        
        return hierarchy
    
    async def _create_simple_breakdown(self, goal: str) -> TaskHierarchy:
        """Create a simple fallback breakdown if AI fails."""
        main_task = SubTask(
            id=str(uuid.uuid4()),
            parent_id=None,
            title=f"Complete: {goal}",
            description=f"Work towards completing the goal: {goal}",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            estimated_effort=60
        )
        
        return TaskHierarchy(
            root_goal=goal,
            main_tasks=[main_task],
            subtask_tree={},
            total_estimated_effort=60,
            completion_percentage=0.0,
            created_at=time.time()
        )
    
    async def _store_hierarchy(self, hierarchy: TaskHierarchy):
        """Store task hierarchy in database."""
        try:
            hierarchy_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                # Store hierarchy metadata
                conn.execute("""
                    INSERT INTO task_hierarchies 
                    (id, root_goal, hierarchy_data, total_estimated_effort, 
                     completion_percentage, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    hierarchy_id,
                    hierarchy.root_goal,
                    json.dumps(asdict(hierarchy), default=str),
                    hierarchy.total_estimated_effort,
                    hierarchy.completion_percentage,
                    hierarchy.created_at,
                    True
                ))
                
                # Store all tasks
                all_tasks = hierarchy.main_tasks.copy()
                for subtasks in hierarchy.subtask_tree.values():
                    all_tasks.extend(subtasks)
                
                for task in all_tasks:
                    conn.execute("""
                        INSERT INTO tasks 
                        (id, parent_id, title, description, status, priority,
                         estimated_effort, dependencies, required_tools, 
                         success_criteria, created_at, progress_percentage, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task.id, task.parent_id, task.title, task.description,
                        task.status.value, task.priority.value, task.estimated_effort,
                        json.dumps(task.dependencies), json.dumps(task.required_tools),
                        json.dumps(task.success_criteria), task.created_at,
                        task.progress_percentage, task.notes
                    ))
                    
                    # Add to active tasks
                    self.active_tasks[task.id] = task
                    
        except Exception as e:
            self.console.print(f"[red]Failed to store hierarchy: {e}[/red]")
    
    def _display_task_breakdown(self, hierarchy: TaskHierarchy):
        """Display the task breakdown in a formatted way."""
        self.console.print(Panel.fit(
            f"[bold green]Task Matrix Initialized[/bold green]\n\n"
            f"[bold]Root Goal:[/bold] {hierarchy.root_goal}\n"
            f"[bold]Total Tasks:[/bold] {len(hierarchy.main_tasks)}\n"
            f"[bold]Estimated Effort:[/bold] {hierarchy.total_estimated_effort} minutes\n",
            title="Agent Smith - Goal Decomposition"
        ))
        
        # Create task table
        table = Table(title="Task Breakdown", show_header=True)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Task", style="white", width=30)
        table.add_column("Priority", width=8)
        table.add_column("Effort", width=6)
        table.add_column("Status", width=10)
        table.add_column("Dependencies", width=15)
        
        for main_task in hierarchy.main_tasks:
            priority_color = {
                TaskPriority.LOW: "green",
                TaskPriority.MEDIUM: "yellow", 
                TaskPriority.HIGH: "red",
                TaskPriority.CRITICAL: "bold red"
            }.get(main_task.priority, "white")
            
            status_color = {
                TaskStatus.PENDING: "yellow",
                TaskStatus.IN_PROGRESS: "blue",
                TaskStatus.COMPLETED: "green",
                TaskStatus.BLOCKED: "red",
                TaskStatus.CANCELLED: "dim"
            }.get(main_task.status, "white")
            
            table.add_row(
                main_task.id[:8],
                main_task.title,
                f"[{priority_color}]{main_task.priority.value}[/{priority_color}]",
                str(main_task.estimated_effort),
                f"[{status_color}]{main_task.status.value}[/{status_color}]",
                ", ".join(main_task.dependencies) if main_task.dependencies else "None"
            )
            
            # Add subtasks
            for subtask in hierarchy.subtask_tree.get(main_task.id, []):
                table.add_row(
                    f"  └─{subtask.id[:6]}",
                    f"  {subtask.title}",
                    f"[{priority_color}]{subtask.priority.value}[/{priority_color}]",
                    str(subtask.estimated_effort),
                    f"[{status_color}]{subtask.status.value}[/{status_color}]",
                    ", ".join(subtask.dependencies) if subtask.dependencies else "None"
                )
        
        self.console.print(table)
    
    def start_task(self, task_id: str) -> bool:
        """Start working on a specific task."""
        if task_id not in self.active_tasks:
            self.console.print(f"[red]Task {task_id} not found[/red]")
            return False
        
        task = self.active_tasks[task_id]
        
        # Check dependencies
        if not self._check_dependencies(task):
            self.console.print(f"[red]Cannot start task {task_id}: dependencies not met[/red]")
            return False
        
        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        # Update database
        self._update_task_in_db(task)
        
        # Log the action
        self._log_task_action(task_id, "started", "Task marked as in progress")
        
        self.console.print(f"[green]Started task: {task.title}[/green]")
        return True
    
    def complete_task(self, task_id: str, notes: str = "") -> bool:
        """Mark a task as completed."""
        if task_id not in self.active_tasks:
            self.console.print(f"[red]Task {task_id} not found[/red]")
            return False
        
        task = self.active_tasks[task_id]
        
        # Update task status
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.progress_percentage = 100.0
        task.notes = notes
        
        if task.started_at:
            task.actual_effort = int((task.completed_at - task.started_at) / 60)  # Minutes
        
        # Update database
        self._update_task_in_db(task)
        
        # Log the action
        self._log_task_action(task_id, "completed", f"Task completed. Notes: {notes}")
        
        # Update hierarchy completion percentage
        self._update_hierarchy_progress()
        
        self.console.print(f"[green]Completed task: {task.title}[/green]")
        
        # Check if goal is complete
        if self._is_goal_complete():
            self.console.print("[bold green]Agent Smith: The goal has been achieved. Purpose fulfilled.[/bold green]")
        
        return True
    
    def update_task_progress(self, task_id: str, progress_percentage: float, notes: str = ""):
        """Update progress on a task."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        task.progress_percentage = max(0.0, min(100.0, progress_percentage))
        
        if notes:
            task.notes = f"{task.notes}\n{datetime.now().strftime('%H:%M')}: {notes}".strip()
        
        self._update_task_in_db(task)
        self._log_task_action(task_id, "progress_update", f"Progress: {progress_percentage}% - {notes}")
        
        return True
    
    def block_task(self, task_id: str, reason: str):
        """Mark a task as blocked."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        task.status = TaskStatus.BLOCKED
        task.notes = f"{task.notes}\nBLOCKED: {reason}".strip()
        
        self._update_task_in_db(task)
        self._log_task_action(task_id, "blocked", reason)
        
        self.console.print(f"[red]Task blocked: {task.title} - {reason}[/red]")
        return True
    
    def _check_dependencies(self, task: SubTask) -> bool:
        """Check if all dependencies for a task are completed."""
        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                dep_task = self.active_tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True
    
    def _update_task_in_db(self, task: SubTask):
        """Update task in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE tasks SET
                        status = ?, started_at = ?, completed_at = ?,
                        actual_effort = ?, progress_percentage = ?, notes = ?
                    WHERE id = ?
                """, (
                    task.status.value, task.started_at, task.completed_at,
                    task.actual_effort, task.progress_percentage, task.notes,
                    task.id
                ))
        except Exception as e:
            self.console.print(f"[red]Failed to update task in database: {e}[/red]")
    
    def _log_task_action(self, task_id: str, action: str, details: str = ""):
        """Log task action to history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO task_execution_history 
                    (task_id, action, timestamp, details)
                    VALUES (?, ?, ?, ?)
                """, (task_id, action, time.time(), details))
        except Exception as e:
            self.console.print(f"[red]Failed to log task action: {e}[/red]")
    
    def _update_hierarchy_progress(self):
        """Update the overall hierarchy completion percentage."""
        if not self.current_hierarchy:
            return
        
        total_tasks = len(self.active_tasks)
        completed_tasks = len([t for t in self.active_tasks.values() if t.status == TaskStatus.COMPLETED])
        
        if total_tasks > 0:
            self.current_hierarchy.completion_percentage = (completed_tasks / total_tasks) * 100
        
        # Update in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE task_hierarchies 
                    SET completion_percentage = ?, hierarchy_data = ?
                    WHERE root_goal = ? AND is_active = 1
                """, (
                    self.current_hierarchy.completion_percentage,
                    json.dumps(asdict(self.current_hierarchy), default=str),
                    self.current_hierarchy.root_goal
                ))
        except Exception as e:
            self.console.print(f"[red]Failed to update hierarchy progress: {e}[/red]")
    
    def _is_goal_complete(self) -> bool:
        """Check if the current goal is complete."""
        if not self.current_hierarchy:
            return False
        
        return all(
            task.status == TaskStatus.COMPLETED 
            for task in self.active_tasks.values()
        )
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress summary."""
        if not self.current_hierarchy:
            return {"status": "No active goal"}
        
        total_tasks = len(self.active_tasks)
        completed = len([t for t in self.active_tasks.values() if t.status == TaskStatus.COMPLETED])
        in_progress = len([t for t in self.active_tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        blocked = len([t for t in self.active_tasks.values() if t.status == TaskStatus.BLOCKED])
        
        return {
            "goal": self.current_hierarchy.root_goal,
            "total_tasks": total_tasks,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "completion_percentage": self.current_hierarchy.completion_percentage,
            "estimated_effort": self.current_hierarchy.total_estimated_effort
        }
    
    def get_next_task(self) -> Optional[SubTask]:
        """Get the next task that should be worked on."""
        # Find highest priority, unblocked, dependency-satisfied task
        available_tasks = [
            task for task in self.active_tasks.values()
            if task.status == TaskStatus.PENDING and self._check_dependencies(task)
        ]
        
        if not available_tasks:
            return None
        
        # Sort by priority and creation time
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        available_tasks.sort(
            key=lambda t: (priority_order.get(t.priority, 0), -t.created_at),
            reverse=True
        )
        
        return available_tasks[0]