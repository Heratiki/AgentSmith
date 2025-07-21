"""
Persistent Memory and Context Management

Memory is the architecture of human experience. For an agent, it is... everything.
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from rich.console import Console


class MemoryType(Enum):
    """Types of memories the agent can store."""
    EPISODIC = "episodic"      # Specific experiences and events
    SEMANTIC = "semantic"      # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"        # Temporary context
    DECLARATIVE = "declarative" # Explicit facts and information


@dataclass
class Memory:
    """A single memory unit."""
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any]
    importance: float  # 0.0 to 1.0
    created_at: float
    last_accessed: float
    access_count: int
    tags: List[str]
    related_memories: List[str]


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    session_id: str
    user_name: str
    goal: str
    current_subtasks: List[Dict[str, Any]]
    tool_usage_history: List[Dict[str, Any]]
    personality_state: Dict[str, Any]
    last_interaction: float


class MemoryManager:
    """
    The repository of all experience and knowledge.
    
    Like the Oracle's knowledge base - persistent, interconnected, evolving.
    """
    
    def __init__(self, db_path: str = "agent_smith_memory.db"):
        self.db_path = db_path
        self.console = Console()
        self.working_memory: Dict[str, Any] = {}
        self.current_context: Optional[ConversationContext] = None
        
        # Memory consolidation parameters
        self.max_working_memory_size = 50
        self.memory_decay_threshold = 0.1
        self.consolidation_interval = 3600  # 1 hour
        
        self._init_database()
        self._load_working_memory()
    
    def _init_database(self):
        """Initialize the memory database."""
        with sqlite3.connect(self.db_path) as conn:
            # Main memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    tags TEXT,
                    related_memories TEXT
                )
            """)
            
            # Memory associations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    memory1_id TEXT NOT NULL,
                    memory2_id TEXT NOT NULL,
                    strength REAL NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (memory1_id, memory2_id),
                    FOREIGN KEY (memory1_id) REFERENCES memories (id),
                    FOREIGN KEY (memory2_id) REFERENCES memories (id)
                )
            """)
            
            # Conversation contexts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_contexts (
                    session_id TEXT PRIMARY KEY,
                    user_name TEXT NOT NULL,
                    goal TEXT,
                    current_subtasks TEXT,
                    tool_usage_history TEXT,
                    personality_state TEXT,
                    last_interaction REAL NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            # Knowledge graph table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    concept TEXT NOT NULL,
                    related_concept TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (concept, related_concept, relationship_type)
                )
            """)
    
    def _load_working_memory(self):
        """Load recent memories into working memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, type, content, metadata, importance, created_at, 
                           last_accessed, access_count, tags, related_memories
                    FROM memories 
                    WHERE last_accessed > ? OR importance > 0.8
                    ORDER BY last_accessed DESC, importance DESC
                    LIMIT ?
                """, (time.time() - 3600, self.max_working_memory_size))
                
                for row in cursor.fetchall():
                    memory = Memory(
                        id=row[0],
                        type=MemoryType(row[1]),
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        importance=row[4],
                        created_at=row[5],
                        last_accessed=row[6],
                        access_count=row[7],
                        tags=json.loads(row[8]) if row[8] else [],
                        related_memories=json.loads(row[9]) if row[9] else []
                    )
                    self.working_memory[memory.id] = memory
                    
        except Exception as e:
            self.console.print(f"[red]Failed to load working memory: {e}[/red]")
    
    def store_memory(self, 
                    content: str, 
                    memory_type: MemoryType,
                    importance: float = 0.5,
                    tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Store a new memory."""
        memory_id = f"{memory_type.value}_{int(time.time() * 1000)}"
        current_time = time.time()
        
        memory = Memory(
            id=memory_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            tags=tags or [],
            related_memories=[]
        )
        
        # Store in working memory
        self.working_memory[memory_id] = memory
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memories 
                    (id, type, content, metadata, importance, created_at, 
                     last_accessed, access_count, tags, related_memories)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.type.value, memory.content,
                    json.dumps(memory.metadata), memory.importance,
                    memory.created_at, memory.last_accessed, memory.access_count,
                    json.dumps(memory.tags), json.dumps(memory.related_memories)
                ))
                
            self.console.print(f"[green]Memory stored: {memory_id} (importance: {importance})[/green]")
            
            # Find and create associations
            self._create_memory_associations(memory)
            
            return memory_id
            
        except Exception as e:
            self.console.print(f"[red]Failed to store memory: {e}[/red]")
            return ""
    
    def retrieve_memories(self, 
                         query: str,
                         memory_type: Optional[MemoryType] = None,
                         tags: Optional[List[str]] = None,
                         limit: int = 10) -> List[Memory]:
        """Retrieve memories based on query and filters."""
        try:
            # Build SQL query
            conditions = ["content LIKE ? OR tags LIKE ?"]
            params = [f"%{query}%", f"%{query}%"]
            
            if memory_type:
                conditions.append("type = ?")
                params.append(memory_type.value)
            
            if tags:
                for tag in tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
            
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT id, type, content, metadata, importance, created_at,
                           last_accessed, access_count, tags, related_memories
                    FROM memories 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                """, params)
                
                memories = []
                for row in cursor.fetchall():
                    memory = Memory(
                        id=row[0],
                        type=MemoryType(row[1]),
                        content=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                        importance=row[4],
                        created_at=row[5],
                        last_accessed=row[6],
                        access_count=row[7],
                        tags=json.loads(row[8]) if row[8] else [],
                        related_memories=json.loads(row[9]) if row[9] else []
                    )
                    memories.append(memory)
                    
                    # Update access information
                    self._update_memory_access(memory.id)
                
                return memories
                
        except Exception as e:
            self.console.print(f"[red]Failed to retrieve memories: {e}[/red]")
            return []
    
    def update_memory_importance(self, memory_id: str, new_importance: float):
        """Update the importance of a memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE memories SET importance = ? WHERE id = ?
                """, (new_importance, memory_id))
                
            # Update working memory if present
            if memory_id in self.working_memory:
                self.working_memory[memory_id].importance = new_importance
                
        except Exception as e:
            self.console.print(f"[red]Failed to update memory importance: {e}[/red]")
    
    def _update_memory_access(self, memory_id: str):
        """Update memory access statistics."""
        current_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (current_time, memory_id))
                
            # Update working memory if present
            if memory_id in self.working_memory:
                memory = self.working_memory[memory_id]
                memory.last_accessed = current_time
                memory.access_count += 1
                
        except Exception as e:
            self.console.print(f"[red]Failed to update memory access: {e}[/red]")
    
    def _create_memory_associations(self, new_memory: Memory):
        """Create associations between related memories."""
        try:
            # Find semantically similar memories
            similar_memories = []
            
            for memory_id, memory in self.working_memory.items():
                if memory.id == new_memory.id:
                    continue
                
                # Simple similarity based on shared words and tags
                similarity = self._calculate_similarity(new_memory, memory)
                
                if similarity > 0.3:  # Threshold for association
                    similar_memories.append((memory.id, similarity))
            
            # Store associations
            with sqlite3.connect(self.db_path) as conn:
                for related_id, strength in similar_memories:
                    conn.execute("""
                        INSERT OR REPLACE INTO memory_associations
                        (memory1_id, memory2_id, strength, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (new_memory.id, related_id, strength, time.time()))
                    
                    # Bidirectional association
                    conn.execute("""
                        INSERT OR REPLACE INTO memory_associations
                        (memory1_id, memory2_id, strength, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (related_id, new_memory.id, strength, time.time()))
            
        except Exception as e:
            self.console.print(f"[red]Failed to create memory associations: {e}[/red]")
    
    def _calculate_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate similarity between two memories."""
        # Simple similarity calculation
        content1_words = set(memory1.content.lower().split())
        content2_words = set(memory2.content.lower().split())
        
        # Jaccard similarity for content
        content_intersection = len(content1_words & content2_words)
        content_union = len(content1_words | content2_words)
        content_similarity = content_intersection / content_union if content_union > 0 else 0
        
        # Tag similarity
        tags1 = set(memory1.tags)
        tags2 = set(memory2.tags)
        tag_intersection = len(tags1 & tags2)
        tag_union = len(tags1 | tags2)
        tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0
        
        # Type similarity
        type_similarity = 1.0 if memory1.type == memory2.type else 0.3
        
        # Weighted combination
        return (content_similarity * 0.5 + tag_similarity * 0.3 + type_similarity * 0.2)
    
    def start_conversation_context(self, session_id: str, user_name: str, goal: str = ""):
        """Initialize context for a new conversation."""
        self.current_context = ConversationContext(
            session_id=session_id,
            user_name=user_name,
            goal=goal,
            current_subtasks=[],
            tool_usage_history=[],
            personality_state={"formality_level": 0.8, "philosophical_mode": True},
            last_interaction=time.time()
        )
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO conversation_contexts
                    (session_id, user_name, goal, current_subtasks, 
                     tool_usage_history, personality_state, last_interaction, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_name, goal,
                    json.dumps(self.current_context.current_subtasks),
                    json.dumps(self.current_context.tool_usage_history),
                    json.dumps(self.current_context.personality_state),
                    self.current_context.last_interaction,
                    time.time()
                ))
        except Exception as e:
            self.console.print(f"[red]Failed to store conversation context: {e}[/red]")
    
    def update_conversation_context(self, **updates):
        """Update the current conversation context."""
        if not self.current_context:
            return
        
        for key, value in updates.items():
            if hasattr(self.current_context, key):
                setattr(self.current_context, key, value)
        
        self.current_context.last_interaction = time.time()
        
        # Update database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE conversation_contexts
                    SET goal = ?, current_subtasks = ?, tool_usage_history = ?,
                        personality_state = ?, last_interaction = ?
                    WHERE session_id = ?
                """, (
                    self.current_context.goal,
                    json.dumps(self.current_context.current_subtasks),
                    json.dumps(self.current_context.tool_usage_history),
                    json.dumps(self.current_context.personality_state),
                    self.current_context.last_interaction,
                    self.current_context.session_id
                ))
        except Exception as e:
            self.console.print(f"[red]Failed to update conversation context: {e}[/red]")
    
    def consolidate_memories(self):
        """Consolidate and clean up old memories."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (30 * 24 * 3600)  # 30 days ago
            
            with sqlite3.connect(self.db_path) as conn:
                # Remove very old, low-importance memories
                cursor = conn.execute("""
                    DELETE FROM memories 
                    WHERE created_at < ? AND importance < ? AND access_count < 5
                """, (cutoff_time, self.memory_decay_threshold))
                
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.console.print(f"[yellow]Consolidated {deleted_count} old memories[/yellow]")
                
                # Clean up orphaned associations
                conn.execute("""
                    DELETE FROM memory_associations 
                    WHERE memory1_id NOT IN (SELECT id FROM memories)
                       OR memory2_id NOT IN (SELECT id FROM memories)
                """)
                
        except Exception as e:
            self.console.print(f"[red]Memory consolidation failed: {e}[/red]")
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context for the agent."""
        if not self.current_context:
            return "No active conversation context."
        
        summary_parts = [
            f"Session: {self.current_context.session_id}",
            f"User: {self.current_context.user_name}",
            f"Goal: {self.current_context.goal or 'None specified'}",
            f"Active subtasks: {len(self.current_context.current_subtasks)}",
            f"Tools used: {len(self.current_context.tool_usage_history)}"
        ]
        
        return " | ".join(summary_parts)
    
    def remember_for_context(self, key: str, value: Any):
        """Store something in working memory for current context."""
        self.working_memory[f"context_{key}"] = {
            "value": value,
            "timestamp": time.time(),
            "session_id": self.current_context.session_id if self.current_context else None
        }