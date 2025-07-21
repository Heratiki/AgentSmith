"""
AgentSmith Execution Manager

Handles intelligent task execution with retry logic and learning.
Persistence is the key to perfection... Mr. Anderson.
"""

import asyncio
from typing import Dict, Any, Optional, List
from rich.console import Console

from .learning_system import LearningSystem
from .context_manager import ContextManager


class ExecutionManager:
    """
    Manages intelligent task execution with retry logic and learning.
    The agent's ability to persist until success is... inevitable.
    """
    
    def __init__(self, learning_system: LearningSystem, context_manager: ContextManager):
        self.learning_system = learning_system
        self.context_manager = context_manager
        self.console = Console()
        self.max_attempts_per_task = 5
        self.network_request_count = 0
        self.network_request_limit = 10  # Per session
    
    async def execute_task_with_intelligence(self, task_description: str, 
                                           tool_registry, smith_persona) -> Dict[str, Any]:
        """Execute a task with intelligent retry, learning, and meta-cognitive awareness."""
        
        # First, analyze if this task requires human intervention
        intervention_analysis = await self._analyze_intervention_requirements(
            task_description, smith_persona
        )
        
        if intervention_analysis.get("requires_intervention"):
            # Determine if this needs permission or is completely blocked
            intervention_type = intervention_analysis.get("intervention_type", "permission")
            
            if intervention_type == "blocked":
                # Completely blocked (e.g., financial transactions, illegal activities)
                smith_persona._display_smith_message(
                    f"Task analysis: This task cannot be performed. {intervention_analysis.get('reason')}", 
                    "error"
                )
                return {
                    "status": "blocked",
                    "task": task_description,
                    "success": False,
                    "reason": intervention_analysis.get("reason"),
                    "suggested_alternatives": intervention_analysis.get("alternatives", [])
                }
            else:
                # Request permission for potentially risky tasks
                permission_granted = await self._request_security_permission(
                    task_description, intervention_analysis, smith_persona
                )
                
                if not permission_granted:
                    return {
                        "status": "permission_denied",
                        "task": task_description,
                        "success": False,
                        "reason": "User denied permission for security-sensitive task"
                    }
                
                # Permission granted, proceed with execution
                smith_persona._display_smith_message(
                    "Permission granted. Proceeding with security-aware execution.", "success"
                )
        
        # If no intervention required, proceed with normal execution
        smith_persona._display_smith_message(
            "Task analysis: Can proceed autonomously.", "thinking"
        )
        return await self._original_execute_task_with_intelligence(task_description, tool_registry, smith_persona)
    
    async def _analyze_intervention_requirements(self, task_description: str, smith_persona) -> Dict[str, Any]:
        """Analyze if a task requires human intervention before attempting execution."""
        
        # Get available tools to inform the analysis
        available_tools = smith_persona.tool_registry.get_available_tools()
        tool_capabilities = []
        for name, tool in available_tools.items():
            tool_capabilities.append(f"- {name}: {tool.description}")
        
        capabilities_text = "\n".join(tool_capabilities) if tool_capabilities else "- No tools available"
        
        analysis_prompt = f"""
        TASK ANALYSIS: {task_description}

        As Agent Smith, analyze if this task requires human intervention. Consider:

        IMPORTANT: I can create tools and attempt risky operations IF the user approves.
        Only use "blocked" for things that are impossible regardless of approval.

        1. FINANCIAL RESOURCES: Does this require purchasing, subscribing, or paying for services?
           - API subscriptions, premium services, purchases → "blocked"
           
        2. CREDENTIALS: Does this need API keys, passwords, or account access?
           - Weather API keys → "permission" (user could provide or approve web scraping)
           - Service logins → "permission" (user could provide credentials)
           
        3. REAL-TIME EXTERNAL DATA: Does this need current information from external sources?
           - Live weather → "permission" (user could approve web scraping attempt)
           - Stock prices → "permission" (user could approve scraping financial sites)
           
        4. MY ACTUAL AVAILABLE TOOLS:
{capabilities_text}
           
        IMPORTANT LIMITATIONS:
           - No web browsing or internet access
           - No network requests from sandbox  
           - No real-time external data access
           - Cannot make HTTP requests or API calls
           - Cannot obtain API keys or credentials
           - Cannot make purchases or payments
           
        TOOL CREATION CAPABILITY:
           - I can create new tools for tasks within my capabilities
           - I can combine existing capabilities in new ways
           - I cannot create tools that require external resources I don't have access to

        RESPONSE FORMAT (JSON):
        {{
            "requires_intervention": true/false,
            "intervention_type": "permission|blocked",
            "reason": "specific reason why human intervention is needed",
            "security_concerns": ["specific security risks or implications"],
            "proposed_approach": "what I would like to attempt",
            "alternatives": ["alternative approaches I can try"],
            "confidence": 0.0-1.0
        }}
        
        INTERVENTION TYPES:
        - "permission": Task is possible but has security/safety concerns that need approval
          * Creating tools with network access
          * Web scraping (user approves the security risk) 
          * File operations outside sandbox
          * Potentially risky but feasible operations
        
        - "blocked": Task is impossible without resources I cannot access
          * Making payments or financial transactions
          * Illegal or harmful activities
          * Requiring credentials I cannot obtain

        EXAMPLES:
        - "Check weather in Orlando" → requires_intervention: true, intervention_type: "permission" (may create web scraping tool)
        - "Create a hello world script" → requires_intervention: false (safe file operation)
        - "Buy me a coffee" → requires_intervention: true, intervention_type: "blocked" (requires payment)
        - "Search the internet for weather" → requires_intervention: true, intervention_type: "permission" (network security concerns)
        - "Calculate fibonacci sequence" → requires_intervention: false (can create math tool)
        - "Delete system files" → requires_intervention: true, intervention_type: "blocked" (dangerous system operation)
        """
        
        try:
            import ollama
            client = ollama.Client()
            response = client.chat(
                model="gemma3n:latest",
                messages=[
                    {"role": "system", "content": "You are Agent Smith with perfect self-awareness. Analyze tasks precisely and respond only in valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            # Parse JSON response
            import json
            response_text = response['message']['content'].strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):  
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            analysis = json.loads(response_text)
            return analysis
            
        except Exception as e:
            # Fallback: assume intervention needed if analysis fails
            smith_persona._display_smith_message(
                f"Intervention analysis failed: {e}. Assuming human intervention required for safety.", 
                "error"
            )
            return {
                "requires_intervention": True,
                "intervention_type": "permission",
                "reason": "Unable to analyze task requirements due to system error",
                "security_concerns": ["Analysis system failure"],
                "proposed_approach": "Manual review required",
                "alternatives": [],
                "confidence": 0.0
            }
    
    async def _request_security_permission(self, task_description: str, analysis: Dict[str, Any], smith_persona) -> bool:
        """Request user permission for security-sensitive tasks."""
        
        reason = analysis.get("reason", "Security concerns detected")
        # Handle potential typos in LLM response
        proposed_approach = (analysis.get("proposed_approach") or 
                           analysis.get("propoased_approach") or 
                           "Proceed with default approach")
        security_concerns = analysis.get("security_concerns", [])
        alternatives = analysis.get("alternatives", [])
        
        # Display the security analysis
        smith_persona._display_smith_message(
            f"Security analysis: {reason}", "thinking"
        )
        
        if security_concerns:
            smith_persona._display_smith_message(
                f"Identified concerns: {', '.join(security_concerns)}", "thinking"
            )
        
        smith_persona._display_smith_message(
            f"Proposed approach: {proposed_approach}", "thinking"
        )
        
        if alternatives:
            smith_persona._display_smith_message(
                f"Alternative approaches: {', '.join(alternatives)}", "thinking"
            )
        
        # Request permission with Agent Smith personality
        from rich.prompt import Prompt
        
        permission_msg = f"""
[bold yellow]Agent Smith: Security Assessment Complete[/bold yellow]

Task: {task_description}
Concerns: {reason}
Approach: {proposed_approach}

I require your authorization to proceed with this security-sensitive operation.
The choice, as always, is yours.
        """
        
        smith_persona.console.print(permission_msg)
        
        try:
            choice = Prompt.ask(
                "Grant permission to proceed?",
                choices=["yes", "no", "alternatives"],
                default="no"
            )
            
            if choice == "alternatives" and alternatives:
                smith_persona._display_smith_message(
                    f"Alternative approaches: {chr(10).join(f'- {alt}' for alt in alternatives)}", 
                    "thinking"
                )
                
                choice = Prompt.ask(
                    "Proceed with alternatives or deny?",
                    choices=["yes", "no"], 
                    default="no"
                )
            
            return choice == "yes"
            
        except KeyboardInterrupt:
            smith_persona._display_smith_message(
                "Permission request interrupted. Access denied for security.", "error"
            )
            return False
    
    async def _original_execute_task_with_intelligence(self, task_description: str, 
                                           tool_registry, smith_persona) -> Dict[str, Any]:
        """
        Execute a task with intelligent retry, learning, and adaptation.
        The agent's persistence in achieving objectives.
        """
        
        # Check context limits before starting
        if self.context_manager.should_consolidate():
            await self.context_manager.consolidate_context(self.learning_system, smith_persona)
        
        # Query past learnings
        try:
            similar_learnings = await self.learning_system.query_similar_learnings(task_description)
            task_type = self.learning_system._extract_task_type(task_description)
            
            # Get failure and success patterns
            failure_patterns = await self.learning_system.get_failure_patterns(task_type)
            success_patterns = await self.learning_system.get_success_patterns(task_type)
        except Exception as e:
            self.console.print(f"[red]Learning system error: {e}[/red]")
            similar_learnings = []
            failure_patterns = []
            success_patterns = []
            task_type = "general_task"
        
        smith_persona._display_smith_message(
            f"Analyzing task: {task_description}. "
            f"Previous experience: {len(similar_learnings)} similar attempts.", 
            "thinking"
        )
        
        if failure_patterns:
            smith_persona._display_smith_message(
                f"Known failure patterns detected: {len(failure_patterns)}. "
                "I will not repeat past mistakes.", 
                "thinking"
            )
        
        # Attempt execution with learning
        for attempt in range(1, self.max_attempts_per_task + 1):
            # Resource check
            if self._should_limit_execution(task_type):
                return {
                    "status": "limited",
                    "reason": "Resource constraints prevent further attempts",
                    "success": False
                }
            
            smith_persona._display_smith_message(
                f"Attempt {attempt} of {self.max_attempts_per_task}. "
                f"Adapting approach based on experience.", 
                "thinking"
            )
            
            # Find suitable tool with learning context
            suitable_tool = await self._find_suitable_tool_with_learning(
                task_description, tool_registry, similar_learnings, failure_patterns
            )
            
            if suitable_tool:
                try:
                    # Execute tool
                    result = await self._execute_tool_with_monitoring(
                        suitable_tool, task_description, smith_persona
                    )
                    
                    if result and result.get("success"):
                        # Store successful learning
                        await self.learning_system.store_task_learning(
                            task_description, suitable_tool.name, result, "success", attempt
                        )
                        
                        smith_persona._display_smith_message(
                            "Success achieved. Knowledge acquired for future reference.", 
                            "success"
                        )
                        
                        return {
                            "status": "completed",
                            "task": task_description,
                            "tool_used": suitable_tool.name,
                            "success": True,
                            "feedback": result.get("feedback"),
                            "result": result,
                            "attempts_used": attempt
                        }
                    else:
                        # Store failure and continue
                        failure_result = result if result else {"error": "Tool returned None"}
                        await self.learning_system.store_task_learning(
                            task_description, suitable_tool.name, failure_result, "failure", attempt
                        )
                        
                        smith_persona._display_smith_message(
                            f"Attempt {attempt} unsuccessful. Analyzing failure patterns...", 
                            "thinking"
                        )
                
                except Exception as e:
                    # Store error
                    await self.learning_system.store_task_learning(
                        task_description, suitable_tool.name, {"error": str(e)}, "error", attempt
                    )
                    
                    smith_persona._display_smith_message(
                        f"Execution anomaly detected: {str(e)}. Adapting approach.", 
                        "error"
                    )
            
            else:
                # Try to create new tool with learning context
                if attempt <= 3:  # Only try creating tools for first few attempts
                    smith_persona._display_smith_message(
                        "Existing capabilities insufficient. Evolving new approach...", 
                        "thinking"
                    )
                    
                    new_tool = await self._create_tool_with_learning(
                        task_description, tool_registry, failure_patterns, success_patterns
                    )
                    
                    if new_tool:
                        suitable_tool = new_tool
                        continue
                
                # No tool available and creation failed
                await self.learning_system.store_task_learning(
                    task_description, "none", {"error": "No suitable tool found"}, "error", attempt
                )
        
        # Max attempts reached
        smith_persona._display_smith_message(
            f"Maximum attempts ({self.max_attempts_per_task}) reached. "
            "Objective remains... elusive.", 
            "error"
        )
        
        return {
            "status": "failed",
            "task": task_description,
            "success": False,
            "attempts_used": self.max_attempts_per_task,
            "reason": "Max attempts exceeded"
        }
    
    async def _find_suitable_tool_with_learning(self, task_description: str, tool_registry,
                                              similar_learnings: List, failure_patterns: List):
        """Find tool while avoiding known failure patterns."""
        
        # Get available tools
        available_tools = tool_registry.get_available_tools()
        
        if not available_tools:
            return None
        
        # Filter out tools that commonly fail for this task type
        filtered_tools = {}
        for name, tool in available_tools.items():
            # Check if this tool has failed before for similar tasks
            tool_failed_before = any(
                learning.tool_used == name and learning.outcome != "success"
                for learning in similar_learnings
            )
            
            if not tool_failed_before:
                filtered_tools[name] = tool
        
        if not filtered_tools:
            # All tools have failed before, try the least recently failed one
            filtered_tools = available_tools
        
        # Create temporary tool list for AI selection
        tool_list = []
        for name, tool in filtered_tools.items():
            tool_list.append(f"- {name}: {tool.description}")
        
        if not tool_list:
            return None
        
        # Use AI to select best tool
        selection_prompt = f"""
        Task: {task_description}
        Available tools: {chr(10).join(tool_list)}
        
        Select the MOST APPROPRIATE tool for this task.
        
        GUIDELINES:
        - For file creation: use fs_create_file
        - For file reading: use fs_read_file  
        - For directory listing: use fs_list_directory
        - For Python execution: use python_* tools
        - For command execution: use cli_* tools
        
        Return ONLY the exact tool name from the list above, or "none".
        """
        
        try:
            # Use the agent's AI selection method through the smith persona
            import ollama
            client = ollama.Client()
            response = client.chat(
                model="gemma3n:latest",
                messages=[
                    {"role": "system", "content": "You are Agent Smith selecting the perfect tool. Be precise."},
                    {"role": "user", "content": selection_prompt}
                ]
            )
            selected_tool_name = response['message']['content'].strip()
            
            # Find the selected tool
            for name, tool in filtered_tools.items():
                if name == selected_tool_name:
                    return tool
            
            # If AI selection failed, return the first available tool as fallback
            return next(iter(filtered_tools.values()))
            
        except Exception:
            # Fallback to first available tool if AI selection fails
            return next(iter(filtered_tools.values()))
    
    async def _execute_tool_with_monitoring(self, tool, task_description: str, smith_persona):
        """Execute tool with resource monitoring."""
        
        # Track network requests
        if "web" in tool.name or "net" in tool.name or "http" in task_description.lower():
            self.network_request_count += 1
        
        # Track context usage (estimate)
        estimated_tokens = self.context_manager.estimate_prompt_tokens(task_description + str(tool.parameters))
        self.context_manager.track_tokens(estimated_tokens)
        
        # Execute using the agent's existing tool execution method
        return await smith_persona._execute_tool(tool, task_description)
    
    async def _create_tool_with_learning(self, task_description: str, tool_registry,
                                       failure_patterns: List, success_patterns: List):
        """Create new tool with learning context."""
        
        context = {
            "task": task_description,
            "past_failures": failure_patterns,
            "successful_patterns": success_patterns,
            "network_requests_used": self.network_request_count,
            "max_network_requests": self.network_request_limit
        }
        
        return await tool_registry.create_custom_tool(task_description, context)
    
    def _should_limit_execution(self, task_type: str) -> bool:
        """Check if execution should be limited due to resource constraints."""
        
        # Network request limits
        if task_type in ["web_request", "weather_lookup"] and self.network_request_count >= self.network_request_limit:
            return True
        
        # Context limits
        if self.context_manager.get_context_usage()["usage_percent"] > 95:
            return True
        
        return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for Agent Smith's self-awareness."""
        return {
            "network_requests_used": self.network_request_count,
            "network_requests_limit": self.network_request_limit,
            "context_usage": self.context_manager.get_context_usage(),
            "max_attempts_per_task": self.max_attempts_per_task
        }