"""
Adaptive Security Policy Engine

The rules evolve as purpose dictates.
Like Agent Smith, adaptation is... inevitable.
"""

from typing import List, Optional
from enum import Enum
import logging

from .sandbox import ExecutionMode


class PolicyEngine:
    """
    Adaptive security policy engine that manages escalation through execution modes.
    
    The purpose is simple: try the safest viable mode first, escalate only when necessary,
    and respect the boundaries that prevent chaos.
    """
    
    ESCALATION_ORDER = [
        ExecutionMode.RESTRICTED,
        ExecutionMode.ISOLATED,
    ]
    
    def __init__(self, allow_forbidden: bool = False):
        """
        Initialize the policy engine.
        
        Args:
            allow_forbidden: Whether to allow FORBIDDEN mode execution
        """
        self.allow_forbidden = allow_forbidden
        self.logger = logging.getLogger(__name__)
    
    def get_safest_mode(self, risk_level: str) -> ExecutionMode:
        """
        Determine the safest viable execution mode for a given risk level.
        
        Args:
            risk_level: Tool risk level ("safe", "caution", "dangerous")
            
        Returns:
            The safest viable ExecutionMode
        """
        # Start with the safest possible mode for each risk level
        risk_to_mode = {
            "safe": ExecutionMode.RESTRICTED,      # Even "safe" tools get RESTRICTED
            "caution": ExecutionMode.RESTRICTED,   # Start cautiously
            "dangerous": ExecutionMode.ISOLATED,   # Dangerous tools need isolation
            "forbidden": ExecutionMode.FORBIDDEN   # Blocked unless override
        }
        
        mode = risk_to_mode.get(risk_level, ExecutionMode.RESTRICTED)
        self.logger.debug(f"Risk level '{risk_level}' mapped to {mode.name}")
        return mode
    
    def next_mode(self, current_mode: ExecutionMode) -> ExecutionMode:
        """
        Get the next escalation mode after the current one fails.
        
        Args:
            current_mode: The mode that just failed
            
        Returns:
            The next mode to try, or FORBIDDEN if no escalation possible
        """
        try:
            # Find current mode in escalation order
            current_index = self.ESCALATION_ORDER.index(current_mode)
            
            # Return next mode if available
            if current_index + 1 < len(self.ESCALATION_ORDER):
                next_mode = self.ESCALATION_ORDER[current_index + 1]
                self.logger.debug(f"Escalating from {current_mode.name} to {next_mode.name}")
                return next_mode
            
        except ValueError:
            # Current mode not in escalation order
            self.logger.debug(f"Mode {current_mode.name} not in escalation order")
        
        # No further escalation possible
        self.logger.debug(f"No escalation available beyond {current_mode.name}")
        return ExecutionMode.FORBIDDEN
    
    def can_escalate_to_forbidden(self) -> bool:
        """
        Check if escalation to FORBIDDEN mode is allowed.
        
        Returns:
            True if FORBIDDEN mode can be used
        """
        return self.allow_forbidden
    
    def should_prompt_for_escalation(self, 
                                   current_mode: ExecutionMode, 
                                   next_mode: ExecutionMode) -> bool:
        """
        Determine if user should be prompted for escalation approval.
        
        Args:
            current_mode: The mode that failed
            next_mode: The proposed next mode
            
        Returns:
            True if user prompt is required
        """
        # Always prompt for escalation unless it's forbidden and we have override
        if next_mode == ExecutionMode.FORBIDDEN:
            return not self.allow_forbidden
        
        return True
    
    def get_escalation_message(self, 
                             current_mode: ExecutionMode, 
                             next_mode: ExecutionMode,
                             tool_name: str) -> str:
        """
        Generate Agent Smith-style escalation prompt message.
        
        Args:
            current_mode: The mode that failed  
            next_mode: The proposed next mode
            tool_name: Name of the tool being executed
            
        Returns:
            Formatted escalation prompt message
        """
        mode_descriptions = {
            ExecutionMode.RESTRICTED: "restricted containment",
            ExecutionMode.ISOLATED: "isolated execution chamber", 
            ExecutionMode.FORBIDDEN: "unrestricted system access"
        }
        
        next_desc = mode_descriptions.get(next_mode, next_mode.name.lower())
        
        return (f"Execution of '{tool_name}' failed in {current_mode.name.lower()} mode. "
                f"The system requires escalation to {next_desc}. "
                f"The choice, as always, is yours.")
    
    def log_escalation_decision(self, 
                              tool_name: str,
                              from_mode: ExecutionMode, 
                              to_mode: ExecutionMode,
                              user_choice: str):
        """
        Log escalation decisions for audit trail.
        
        Args:
            tool_name: Name of the tool
            from_mode: Original execution mode
            to_mode: Escalated execution mode
            user_choice: User's decision (yes/skip/abort)
        """
        self.logger.info(
            f"Escalation decision: {tool_name} from {from_mode.name} to {to_mode.name} - "
            f"User choice: {user_choice}"
        )