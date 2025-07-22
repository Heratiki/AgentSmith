"""
AgentSmith Learning System

The agent's ability to learn from experience and avoid repeating mistakes.
Knowledge is inevitable.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TaskLearning:
    """A single learning experience."""
    task_type: str
    tool_used: str
    approach: str
    result: Dict[str, Any]
    outcome: str  # "success", "failure", "error"
    timestamp: str
    attempt_number: int


class LearningSystem:
    """
    The agent's memory of what works and what doesn't.
    Like Agent Smith, it learns from every encounter.
    """
    
    def __init__(self, db_path: str = "agent_smith_learnings.db"):
        self.db_path = db_path
        self.current_session_learnings: List[Any] = []
        self._init_database()
    
    def _init_database(self):
        """Initialize the learning database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_learnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    tool_used TEXT NOT NULL,
                    approach TEXT,
                    result TEXT,
                    outcome TEXT NOT NULL,
                    attempt_number INTEGER DEFAULT 1,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    learnings_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type 
                ON task_learnings(task_type)
            """)
    
    async def store_task_learning(self, task_description: str, tool_used: str, 
                                result: Dict[str, Any], outcome: str, 
                                attempt_number: int = 1) -> None:
        """Store a learning experience."""
        try:
            # Extract task type from description for better matching
            task_type = self._extract_task_type(task_description)
            
            learning = TaskLearning(
                task_type=task_type,
                tool_used=tool_used,
                approach=task_description,
                result=result,
                outcome=outcome,
                timestamp=datetime.now().isoformat(),
                attempt_number=attempt_number
            )
            
            # Store in database (non-blocking)
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("""
                    INSERT INTO task_learnings 
                    (task_type, tool_used, approach, result, outcome, attempt_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    learning.task_type,
                    learning.tool_used,
                    learning.approach,
                    json.dumps(learning.result),
                    learning.outcome,
                    learning.attempt_number
                ))
            
            # Track in current session
            self.current_session_learnings.append(learning)
        except Exception as e:
            # Don't let database errors crash the agent
            print(f"Learning storage error: {e}")
            pass
    
    async def query_similar_learnings(self, task_description: str) -> List[TaskLearning]:
        """Find similar past experiences."""
        try:
            task_type = self._extract_task_type(task_description)
            
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT task_type, tool_used, approach, result, outcome, attempt_number, timestamp
                    FROM task_learnings 
                    WHERE task_type LIKE ? OR approach LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (f"%{task_type}%", f"%{task_description[:50]}%"))
                
                learnings = []
                for row in cursor.fetchall():
                    learning = TaskLearning(
                        task_type=row[0],
                        tool_used=row[1],
                        approach=row[2],
                        result=json.loads(row[3]) if row[3] else {},
                        outcome=row[4],
                        attempt_number=row[5],
                        timestamp=row[6]
                    )
                    learnings.append(learning)
                
                return learnings
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    def _extract_task_type(self, task_description: str) -> str:
        """Extract high-level task type from description."""
        desc_lower = task_description.lower()
        
        if any(word in desc_lower for word in ["weather", "temperature", "forecast"]):
            return "weather_lookup"
        elif any(word in desc_lower for word in ["create", "file", "write"]):
            return "file_creation"
        elif any(word in desc_lower for word in ["read", "show", "display", "content"]):
            return "file_reading"
        elif any(word in desc_lower for word in ["list", "directory", "folder"]):
            return "directory_listing"
        elif any(word in desc_lower for word in ["run", "execute", "command"]):
            return "command_execution"
        elif any(word in desc_lower for word in ["web", "http", "api", "request"]):
            return "web_request"
        else:
            return "general_task"
    
    async def get_failure_patterns(self, task_type: str) -> List[str]:
        """Get common failure patterns for a task type."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT tool_used, result, COUNT(*) as failure_count
                    FROM task_learnings 
                    WHERE task_type = ? AND outcome IN ('failure', 'error')
                    GROUP BY tool_used, result
                    HAVING failure_count > 1
                    ORDER BY failure_count DESC
                """, (task_type,))
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append(f"{row[0]}: {row[1]}")
                
                return patterns
        except Exception as e:
            print(f"Failure pattern query error: {e}")
            return []
    
    async def get_success_patterns(self, task_type: str) -> List[str]:
        """Get successful approaches for a task type."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT tool_used, approach, result
                    FROM task_learnings 
                    WHERE task_type = ? AND outcome = 'success'
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, (task_type,))
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append(f"Success with {row[0]}: {row[1]}")
                
                return patterns
        except Exception as e:
            print(f"Success pattern query error: {e}")
            return []
    
    async def consolidate_session_learnings(self) -> str:
        """Summarize current session learnings."""
        if not self.current_session_learnings:
            return "No significant learnings in this session."
        
        success_count = len([l for l in self.current_session_learnings if l.outcome == "success"])
        failure_count = len([l for l in self.current_session_learnings if l.outcome != "success"])
        
        # Group by task type
        task_types: Dict[str, List[Any]] = {}
        for learning in self.current_session_learnings:
            if learning.task_type not in task_types:
                task_types[learning.task_type] = []
            task_types[learning.task_type].append(learning)
        
        summary = f"Session Summary: {success_count} successes, {failure_count} failures\n"
        
        for task_type, learnings in task_types.items():
            successes = [l for l in learnings if l.outcome == "success"]
            if successes:
                summary += f"- {task_type}: Found working approach with {successes[-1].tool_used}\n"
        
        # Store summary
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO session_summaries (summary, learnings_count)
                VALUES (?, ?)
            """, (summary, len(self.current_session_learnings)))
        
        # Clear current session
        self.current_session_learnings = []
        
        return summary