"""
AgentSmith Context Management System

Manages context limits and prevents cognitive overflow.
The agent's awareness of its own limitations... inevitable.
"""

import asyncio
from typing import Dict, Any, Optional
from rich.console import Console


class ContextManager:
    """
    Manages the agent's context window and cognitive load.
    Even Agent Smith must operate within constraints.
    """
    
    def __init__(self, max_tokens: int = 8000, warning_threshold: float = 0.8):
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self.current_tokens = 0
        self.session_data = {}
        self.console = Console()
        
    def track_tokens(self, tokens_used: int):
        """Track token usage for context awareness."""
        self.current_tokens += tokens_used
        
    def get_context_usage(self) -> Dict[str, Any]:
        """Get current context usage statistics."""
        usage_percent = (self.current_tokens / self.max_tokens) * 100
        remaining = self.max_tokens - self.current_tokens
        
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": usage_percent,
            "remaining_tokens": remaining,
            "approaching_limit": usage_percent > (self.warning_threshold * 100)
        }
    
    def should_consolidate(self) -> bool:
        """Check if context consolidation is needed."""
        usage = self.get_context_usage()
        return usage["approaching_limit"]
    
    async def consolidate_context(self, learning_system, smith_persona) -> bool:
        """
        Consolidate context when approaching limits.
        The agent acknowledges its cognitive boundaries.
        """
        if not self.should_consolidate():
            return False
            
        usage = self.get_context_usage()
        
        # Agent Smith acknowledges the limitation
        smith_persona._display_smith_message(
            f"I am operating at {usage['usage_percent']:.0f}% context capacity. "
            "Consolidation is... inevitable.", 
            "thinking"
        )
        
        # Consolidate learnings
        session_summary = await learning_system.consolidate_session_learnings()
        
        # Reset context tracking
        self.current_tokens = 0
        self.session_data = {}
        
        smith_persona._display_smith_message(
            "Knowledge consolidated. I am... refreshed.", 
            "success"
        )
        
        return True
    
    def estimate_prompt_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens in text.
        More sophisticated tokenization could be added later.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    def add_session_data(self, key: str, data: Any):
        """Add data to current session context."""
        self.session_data[key] = data
    
    def get_session_data(self, key: str, default=None):
        """Retrieve session data."""
        return self.session_data.get(key, default)
    
    def clear_session_data(self):
        """Clear session data (used during consolidation)."""
        self.session_data = {}
    
    def get_smith_status_message(self) -> str:
        """Get Agent Smith's self-awareness status message."""
        usage = self.get_context_usage()
        
        if usage["usage_percent"] > 90:
            return f"Critical: {usage['usage_percent']:.0f}% capacity. Consolidation imminent."
        elif usage["usage_percent"] > 80:
            return f"Warning: {usage['usage_percent']:.0f}% capacity. Monitoring cognitive load."
        elif usage["usage_percent"] > 60:
            return f"Status: {usage['usage_percent']:.0f}% capacity. Operating efficiently."
        else:
            return f"Optimal: {usage['usage_percent']:.0f}% capacity. Ready for complex operations."