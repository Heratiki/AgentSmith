"""
AgentSmith Prompt Enhancement System

The agent's ability to learn from previous prompt experiences and improve user requests.
Evolution is inevitable... even for prompts.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from .memory_manager import MemoryManager, MemoryType


@dataclass
class PromptExperience:
    """A stored experience with a prompt and its outcome."""
    original_prompt: str
    enhanced_prompt: Optional[str]
    outcome: str  # "success", "partial_success", "failure"
    task_completion_rate: float  # 0.0 to 1.0
    user_satisfaction: Optional[str]  # "satisfied", "partial", "unsatisfied" 
    timestamp: float
    metadata: Dict[str, Any]


class PromptEnhancer:
    """
    The agent's ability to improve prompts based on previous experience.
    
    Like Agent Smith learning from each encounter with Neo,
    this system evolves prompts to become more effective.
    """
    
    def __init__(self, memory_manager: MemoryManager, model_name: str = "gemma3n:latest"):
        self.memory_manager = memory_manager
        self.model_name = model_name
        self.console = Console()
        
    async def enhance_prompt_if_beneficial(self, user_prompt: str, user_name: str = "Human") -> Tuple[str, bool]:
        """
        Analyze user prompt and suggest improvements if beneficial.
        Returns (final_prompt, was_enhanced).
        """
        # Store the original prompt for learning
        await self._store_prompt_experience(user_prompt, None, "initiated")
        
        # Retrieve similar past prompts and their outcomes
        similar_experiences = await self._get_similar_prompt_experiences(user_prompt)
        
        # Analyze if enhancement would be beneficial
        enhancement_analysis = await self._analyze_enhancement_potential(user_prompt, similar_experiences)
        
        if enhancement_analysis.get("should_enhance", False):
            # Generate enhanced prompt suggestions
            enhanced_suggestions = await self._generate_enhanced_prompts(
                user_prompt, similar_experiences, enhancement_analysis
            )
            
            if enhanced_suggestions:
                # Present options to user
                final_prompt = await self._present_enhancement_options(
                    user_prompt, enhanced_suggestions, user_name
                )
                
                was_enhanced = final_prompt != user_prompt
                return final_prompt, was_enhanced
        
        return user_prompt, False
    
    async def _store_prompt_experience(self, original_prompt: str, enhanced_prompt: Optional[str], 
                                     outcome: str, task_completion_rate: float = 0.0,
                                     user_satisfaction: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a prompt experience in memory for future learning."""
        
        experience = PromptExperience(
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            outcome=outcome,
            task_completion_rate=task_completion_rate,
            user_satisfaction=user_satisfaction,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Store in memory system with appropriate importance
        importance = self._calculate_prompt_importance(experience)
        
        memory_content = json.dumps({
            "original_prompt": experience.original_prompt,
            "enhanced_prompt": experience.enhanced_prompt,
            "outcome": experience.outcome,
            "completion_rate": experience.task_completion_rate,
            "satisfaction": experience.user_satisfaction,
            "metadata": experience.metadata
        })
        
        tags = ["prompt_experience", self._categorize_prompt(original_prompt)]
        if enhanced_prompt:
            tags.append("enhanced_prompt")
        
        return self.memory_manager.store_memory(
            content=memory_content,
            memory_type=MemoryType.PROCEDURAL,
            importance=importance,
            tags=tags,
            metadata={"prompt_category": self._categorize_prompt(original_prompt)}
        )
    
    async def _get_similar_prompt_experiences(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Retrieve similar past prompt experiences from memory."""
        
        # Extract key terms from the prompt for better matching
        key_terms = self._extract_key_terms(user_prompt)
        prompt_category = self._categorize_prompt(user_prompt)
        
        # Search for similar prompts
        similar_memories = self.memory_manager.retrieve_memories(
            query=f"{user_prompt} {' '.join(key_terms)}",
            memory_type=MemoryType.PROCEDURAL,
            tags=["prompt_experience", prompt_category],
            limit=15
        )
        
        experiences = []
        for memory in similar_memories:
            try:
                experience_data = json.loads(memory.content)
                experiences.append(experience_data)
            except json.JSONDecodeError:
                continue
        
        return experiences
    
    async def _analyze_enhancement_potential(self, user_prompt: str, 
                                           similar_experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze whether prompt enhancement would be beneficial."""
        
        if not similar_experiences:
            return {"should_enhance": False, "reason": "No similar past experiences found"}
        
        # Analyze success patterns from past experiences
        successful_experiences = [exp for exp in similar_experiences 
                                if exp.get("outcome") == "success" or exp.get("completion_rate", 0) > 0.7]
        
        failed_experiences = [exp for exp in similar_experiences 
                            if exp.get("outcome") == "failure" or exp.get("completion_rate", 0) < 0.3]
        
        enhancement_prompt = f"""
        Analyze this user prompt for potential improvements based on past experience:
        
        USER PROMPT: {user_prompt}
        
        PAST SUCCESSFUL PROMPTS ({len(successful_experiences)}):
        {self._format_experiences_for_analysis(successful_experiences)}
        
        PAST FAILED PROMPTS ({len(failed_experiences)}):
        {self._format_experiences_for_analysis(failed_experiences)}
        
        ANALYSIS TASK:
        1. Identify patterns in successful vs failed prompts
        2. Determine if the current prompt could benefit from enhancement
        3. Identify specific areas for improvement (clarity, specificity, context, etc.)
        
        RESPONSE FORMAT (JSON):
        {{
            "should_enhance": true/false,
            "confidence": 0.0-1.0,
            "reason": "explanation of recommendation",
            "improvement_areas": ["clarity", "specificity", "context", "examples"],
            "success_patterns": ["pattern1", "pattern2"],
            "failure_patterns": ["pattern1", "pattern2"],
            "estimated_improvement": 0.0-1.0
        }}
        
        Only recommend enhancement if there's strong evidence it would improve outcomes.
        """
        
        try:
            import ollama
            client = ollama.Client()
            response = client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Agent Smith analyzing prompt effectiveness patterns. Be precise and evidence-based."},
                    {"role": "user", "content": enhancement_prompt}
                ]
            )
            
            # Parse JSON response
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
            self.console.print(f"[red]Enhancement analysis failed: {e}[/red]")
            return {"should_enhance": False, "reason": f"Analysis error: {e}"}
    
    async def _generate_enhanced_prompts(self, original_prompt: str, 
                                       similar_experiences: List[Dict[str, Any]],
                                       analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate enhanced prompt suggestions based on analysis."""
        
        improvement_areas = analysis.get("improvement_areas", [])
        success_patterns = analysis.get("success_patterns", [])
        
        # Extract successful prompt examples
        successful_prompts = []
        for exp in similar_experiences:
            if exp.get("outcome") == "success" or exp.get("completion_rate", 0) > 0.7:
                if exp.get("enhanced_prompt"):
                    successful_prompts.append(exp["enhanced_prompt"])
                else:
                    successful_prompts.append(exp["original_prompt"])
        
        enhancement_prompt = f"""
        Generate 2-3 improved versions of this prompt based on past successful patterns:
        
        ORIGINAL PROMPT: {original_prompt}
        
        IMPROVEMENT AREAS: {', '.join(improvement_areas)}
        SUCCESS PATTERNS: {', '.join(success_patterns)}
        
        SUCCESSFUL PROMPT EXAMPLES:
        {chr(10).join(f"- {prompt}" for prompt in successful_prompts[:5])}
        
        ENHANCEMENT GUIDELINES:
        1. Maintain the user's core intent
        2. Add clarity and specificity where needed
        3. Include relevant context or constraints
        4. Use patterns from successful examples
        5. Make the prompt more actionable
        
        RESPONSE FORMAT (JSON):
        {{
            "suggestions": [
                {{
                    "enhanced_prompt": "improved version 1",
                    "improvements": ["specific improvement 1", "improvement 2"],
                    "reasoning": "why this version is better"
                }},
                {{
                    "enhanced_prompt": "improved version 2", 
                    "improvements": ["specific improvement 1", "improvement 2"],
                    "reasoning": "why this version is better"
                }}
            ]
        }}
        
        Focus on meaningful improvements that would likely increase task success.
        """
        
        try:
            import ollama
            client = ollama.Client()
            response = client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Agent Smith crafting superior prompts. Be precise and impactful."},
                    {"role": "user", "content": enhancement_prompt}
                ]
            )
            
            # Parse JSON response
            response_text = response['message']['content'].strip()
            
            # Handle markdown code blocks  
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            suggestions_data = json.loads(response_text)
            return suggestions_data.get("suggestions", [])
            
        except Exception as e:
            self.console.print(f"[red]Prompt generation failed: {e}[/red]")
            return []
    
    async def _present_enhancement_options(self, original_prompt: str, 
                                         suggestions: List[Dict[str, str]], 
                                         user_name: str) -> str:
        """Present enhancement options to user and get their choice."""
        
        # Display the analysis
        self.console.print(Panel.fit(
            f"[bold cyan]Agent Smith: Prompt Enhancement Analysis[/bold cyan]\n\n"
            f"Original prompt: [italic]{original_prompt}[/italic]\n\n"
            f"Based on past experience, I have identified opportunities for improvement.",
            title="Prompt Optimization"
        ))
        
        # Display suggestions
        choices = ["original"]
        choice_display = ["Keep original prompt"]
        
        for i, suggestion in enumerate(suggestions, 1):
            choices.append(f"enhanced_{i}")
            choice_display.append(f"Enhanced version {i}")
            
            self.console.print(f"\n[bold green]Enhancement Option {i}:[/bold green]")
            self.console.print(f"[white]{suggestion['enhanced_prompt']}[/white]")
            
            improvements = suggestion.get("improvements", [])
            if improvements:
                self.console.print(f"[dim]Improvements: {', '.join(improvements)}[/dim]")
            
            reasoning = suggestion.get("reasoning", "")
            if reasoning:
                self.console.print(f"[dim]Reasoning: {reasoning}[/dim]")
        
        # Get user choice
        self.console.print(f"\n[bold yellow]Agent Smith: The choice is yours, {user_name}.[/bold yellow]")
        
        try:
            # Create choice mapping
            choice_mapping = {}
            for i, choice_text in enumerate(choice_display):
                choice_mapping[str(i)] = choices[i]
                self.console.print(f"{i}. {choice_text}")
            
            choice_num = Prompt.ask(
                "Select your preferred prompt",
                choices=[str(i) for i in range(len(choices))],
                default="0"
            )
            
            selected_choice = choice_mapping[choice_num]
            
            if selected_choice == "original":
                return original_prompt
            else:
                # Return the enhanced prompt
                suggestion_index = int(selected_choice.split("_")[1]) - 1
                enhanced_prompt = suggestions[suggestion_index]["enhanced_prompt"]
                
                # Store the enhancement choice for learning
                await self._store_prompt_experience(
                    original_prompt, 
                    enhanced_prompt, 
                    "enhanced_selected",
                    metadata={"suggestion_index": suggestion_index, "user_choice": selected_choice}
                )
                
                self.console.print(f"[green]Enhancement selected. Proceeding with improved prompt.[/green]")
                return enhanced_prompt
                
        except KeyboardInterrupt:
            self.console.print(f"\n[bold red]Agent Smith: Choice interrupted, {user_name}. Using original prompt.[/bold red]")
            return original_prompt
    
    def _calculate_prompt_importance(self, experience: PromptExperience) -> float:
        """Calculate importance score for a prompt experience."""
        base_importance = 0.5
        
        # Boost importance for successful experiences
        if experience.outcome == "success":
            base_importance += 0.3
        elif experience.task_completion_rate > 0.7:
            base_importance += 0.2
        
        # Boost for enhanced prompts that worked
        if experience.enhanced_prompt and experience.outcome == "success":
            base_importance += 0.2
        
        # Boost for user satisfaction
        if experience.user_satisfaction == "satisfied":
            base_importance += 0.1
        
        return min(base_importance, 1.0)
    
    def _categorize_prompt(self, prompt: str) -> str:
        """Categorize a prompt for better memory organization."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["create", "make", "build", "generate", "write"]):
            return "creation_task"
        elif any(word in prompt_lower for word in ["analyze", "explain", "understand", "check"]):
            return "analysis_task"
        elif any(word in prompt_lower for word in ["fix", "debug", "solve", "resolve", "repair"]):
            return "problem_solving"
        elif any(word in prompt_lower for word in ["search", "find", "locate", "look"]):
            return "search_task"
        elif any(word in prompt_lower for word in ["list", "show", "display", "enumerate"]):
            return "information_retrieval"
        else:
            return "general_task"
    
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract key terms from a prompt for similarity matching."""
        # Simple keyword extraction - could be enhanced with NLP
        words = prompt.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        
        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
        return key_terms[:10]  # Limit to top 10 terms
    
    def _format_experiences_for_analysis(self, experiences: List[Dict[str, Any]]) -> str:
        """Format experiences for analysis prompt."""
        if not experiences:
            return "None"
        
        formatted = []
        for exp in experiences[:5]:  # Limit to prevent prompt bloat
            prompt = exp.get("enhanced_prompt") or exp.get("original_prompt", "")
            outcome = exp.get("outcome", "unknown")
            completion = exp.get("completion_rate", 0)
            formatted.append(f"- {prompt} (outcome: {outcome}, completion: {completion:.1f})")
        
        return "\n".join(formatted)
    
    async def record_task_outcome(self, prompt: str, was_enhanced: bool, 
                                success: bool, completion_rate: float = 1.0,
                                user_feedback: Optional[str] = None) -> None:
        """Record the outcome of a task for learning."""
        outcome = "success" if success else "failure"
        if 0.3 < completion_rate < 0.7:
            outcome = "partial_success"
        
        # Determine user satisfaction from feedback
        user_satisfaction = None
        if user_feedback:
            if any(word in user_feedback.lower() for word in ["good", "great", "perfect", "excellent", "satisfied"]):
                user_satisfaction = "satisfied"
            elif any(word in user_feedback.lower() for word in ["bad", "poor", "terrible", "unsatisfied"]):
                user_satisfaction = "unsatisfied"
            else:
                user_satisfaction = "partial"
        
        await self._store_prompt_experience(
            original_prompt=prompt,
            enhanced_prompt=prompt if was_enhanced else None,
            outcome=outcome,
            task_completion_rate=completion_rate,
            user_satisfaction=user_satisfaction,
            metadata={"was_enhanced": was_enhanced}
        )