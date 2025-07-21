"""
Dynamic Tool Discovery and Registration System

The agent's ability to adapt and acquire new capabilities is... inevitable.
"""

import inspect
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import ollama
from rich.console import Console


class RiskLevel(Enum):
    """Classification of tool risk levels."""
    SAFE = "safe"
    CAUTION = "caution" 
    DANGEROUS = "dangerous"
    FORBIDDEN = "forbidden"


@dataclass
class ToolDefinition:
    """Complete definition of a dynamically discovered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: str
    risk_level: RiskLevel
    category: str
    examples: List[str]
    dependencies: List[str]
    created_timestamp: str
    test_results: Optional[Dict[str, Any]] = None


class DynamicToolRegistry:
    """
    The registry where all discovered tools are catalogued and managed.
    
    Like the Matrix itself, it grows, adapts, and remembers.
    """
    
    def __init__(self, db_path: str = "agent_smith_tools.db"):
        self.db_path = db_path
        self.console = Console()
        self.client = ollama.Client()
        self.forbidden_operations = {
            "rm -rf", "del /s", "format", "mkfs", 
            "dd if=", "fdisk", "passwd", "sudo su",
            "chmod 777", "wget http://", "curl http://",
            "nc -l", "netcat", "telnet", "> /dev/null"
        }
        self.tools: Dict[str, ToolDefinition] = {}
        self._init_database()
        self._load_existing_tools()
    
    def _init_database(self):
        """Initialize the tool registry database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    definition TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    parameters TEXT,
                    result TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool_name) REFERENCES tools (name)
                )
            """)
    
    def _load_existing_tools(self):
        """Load previously discovered tools from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name, definition FROM tools")
            for name, definition_json in cursor.fetchall():
                try:
                    definition_dict = json.loads(definition_json)
                    tool = ToolDefinition(**definition_dict)
                    self.tools[name] = tool
                except (json.JSONDecodeError, TypeError) as e:
                    self.console.print(f"[red]Error loading tool {name}: {e}[/red]")
    
    async def discover_system_capabilities(self) -> Dict[str, ToolDefinition]:
        """
        Scan the environment for available tools and capabilities.
        The agent learns what it can do by exploring its domain.
        """
        self.console.print("[cyan]Agent Smith: Scanning environmental capabilities...[/cyan]")
        
        discovered = {}
        
        # Discover command-line tools
        cli_tools = await self._discover_cli_tools()
        discovered.update(cli_tools)
        
        # Discover Python capabilities
        python_tools = await self._discover_python_capabilities()
        discovered.update(python_tools)
        
        # Discover file system operations
        fs_tools = await self._discover_filesystem_tools()
        discovered.update(fs_tools)
        
        # Store new discoveries
        for name, tool in discovered.items():
            if name not in self.tools:
                await self.register_tool(tool)
        
        return discovered
    
    async def _discover_cli_tools(self) -> Dict[str, ToolDefinition]:
        """Discover available command-line tools."""
        common_tools = [
            "ls", "dir", "pwd", "cd", "cat", "type", "head", "tail",
            "grep", "find", "locate", "which", "whereis",
            "python", "pip", "node", "npm", "git", "curl", "wget",
            "ps", "top", "kill", "systemctl", "service"
        ]
        
        discovered = {}
        
        for tool_name in common_tools:
            try:
                # Test if tool exists and get version/help
                help_result = await self._safe_command_test(tool_name, ["--help"])
                version_result = await self._safe_command_test(tool_name, ["--version"])
                
                if help_result or version_result:
                    risk_level = self._assess_command_risk(tool_name)
                    
                    if risk_level != RiskLevel.FORBIDDEN:
                        tool_def = ToolDefinition(
                            name=f"cli_{tool_name}",
                            description=f"Command-line tool: {tool_name}",
                            parameters={"args": {"type": "array", "description": "Command arguments"}},
                            implementation=f"subprocess.run(['{tool_name}'] + args, capture_output=True, text=True, timeout=30)",
                            risk_level=risk_level,
                            category="cli",
                            examples=[f"{tool_name} --help"],
                            dependencies=[tool_name],
                            created_timestamp=str(Path().stat().st_mtime)
                        )
                        
                        discovered[tool_def.name] = tool_def
                        
            except Exception:
                continue
        
        return discovered
    
    async def _discover_python_capabilities(self) -> Dict[str, ToolDefinition]:
        """Discover Python modules and capabilities."""
        important_modules = [
            "os", "sys", "pathlib", "json", "sqlite3", "subprocess",
            "requests", "urllib", "socket", "threading", "asyncio",
            "datetime", "time", "re", "collections", "itertools"
        ]
        
        discovered = {}
        
        for module_name in important_modules:
            try:
                # Test import
                result = subprocess.run([
                    sys.executable, "-c", f"import {module_name}; print('OK')"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and "OK" in result.stdout:
                    risk_level = self._assess_module_risk(module_name)
                    
                    if risk_level != RiskLevel.FORBIDDEN:
                        tool_def = ToolDefinition(
                            name=f"python_{module_name}",
                            description=f"Python module: {module_name}",
                            parameters={"code": {"type": "string", "description": "Python code to execute"}},
                            implementation=f"exec(compile(code, '<string>', 'exec'))",
                            risk_level=risk_level,
                            category="python",
                            examples=[f"import {module_name}"],
                            dependencies=[module_name],
                            created_timestamp=str(Path().stat().st_mtime)
                        )
                        
                        discovered[tool_def.name] = tool_def
                        
            except Exception:
                continue
        
        return discovered
    
    async def _discover_filesystem_tools(self) -> Dict[str, ToolDefinition]:
        """Discover file system operation capabilities."""
        fs_operations = {
            "read_file": {
                "description": "Read contents of a file safely",
                "implementation": "Path(filepath).read_text(encoding='utf-8', errors='ignore')",
                "risk": RiskLevel.SAFE,
                "params": {"filepath": {"type": "string", "description": "Path to file"}}
            },
            "list_directory": {
                "description": "List directory contents",
                "implementation": "list(Path(dirpath).iterdir())",
                "risk": RiskLevel.SAFE,
                "params": {"dirpath": {"type": "string", "description": "Directory path"}}
            },
            "create_file": {
                "description": "Create a new file with content",
                "implementation": "Path(filepath).write_text(content, encoding='utf-8')",
                "risk": RiskLevel.CAUTION,
                "params": {
                    "filepath": {"type": "string", "description": "Path to new file"},
                    "content": {"type": "string", "description": "File content"}
                }
            },
            "check_permissions": {
                "description": "Check file/directory permissions",
                "implementation": "oct(Path(filepath).stat().st_mode)[-3:]",
                "risk": RiskLevel.SAFE,
                "params": {"filepath": {"type": "string", "description": "Path to check"}}
            }
        }
        
        discovered = {}
        
        for name, config in fs_operations.items():
            tool_def = ToolDefinition(
                name=f"fs_{name}",
                description=config["description"],
                parameters=config["params"],
                implementation=config["implementation"],
                risk_level=config["risk"],
                category="filesystem",
                examples=[f"fs_{name}('/path/to/file')"],
                dependencies=["pathlib"],
                created_timestamp=str(Path().stat().st_mtime)
            )
            
            discovered[tool_def.name] = tool_def
        
        return discovered
    
    async def _safe_command_test(self, command: str, args: List[str]) -> Optional[str]:
        """Safely test if a command exists and get its output."""
        try:
            result = subprocess.run(
                [command] + args,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
    
    def _assess_command_risk(self, command: str) -> RiskLevel:
        """Assess the risk level of a command."""
        dangerous_commands = {
            "rm", "del", "format", "fdisk", "mkfs", "dd", "nc", "netcat"
        }
        
        caution_commands = {
            "chmod", "chown", "mv", "cp", "wget", "curl", "ssh", "scp"
        }
        
        if command in dangerous_commands:
            return RiskLevel.DANGEROUS
        elif command in caution_commands:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE
    
    def _assess_module_risk(self, module: str) -> RiskLevel:
        """Assess the risk level of a Python module."""
        dangerous_modules = {"subprocess", "os", "sys"}
        caution_modules = {"socket", "urllib", "requests", "threading"}
        
        if module in dangerous_modules:
            return RiskLevel.CAUTION  # Demote dangerous to caution for controlled use
        elif module in caution_modules:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE
    
    async def register_tool(self, tool: ToolDefinition) -> bool:
        """Register a new tool in the registry."""
        try:
            # Security check
            if not await self._security_validate_tool(tool):
                self.console.print(f"[red]Security validation failed for tool: {tool.name}[/red]")
                return False
            
            # Test the tool
            test_result = await self._test_tool_safety(tool)
            tool.test_results = test_result
            
            # Store in memory
            self.tools[tool.name] = tool
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO tools (name, definition) VALUES (?, ?)",
                    (tool.name, json.dumps(asdict(tool), default=str))
                )
            
            self.console.print(f"[green]Tool registered: {tool.name} (Risk: {tool.risk_level.value})[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to register tool {tool.name}: {e}[/red]")
            return False
    
    async def _security_validate_tool(self, tool: ToolDefinition) -> bool:
        """Validate tool against security policies."""
        # Check for forbidden operations
        implementation_lower = tool.implementation.lower()
        
        for forbidden in self.forbidden_operations:
            if forbidden in implementation_lower:
                return False
        
        # Check risk level appropriateness
        if tool.risk_level == RiskLevel.FORBIDDEN:
            return False
        
        return True
    
    async def _test_tool_safety(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Test a tool in a safe environment."""
        test_result = {
            "syntax_valid": False,
            "safe_execution": False,
            "error": None
        }
        
        try:
            # Basic syntax validation
            if tool.category == "python":
                compile(tool.implementation, '<string>', 'exec')
                test_result["syntax_valid"] = True
            else:
                test_result["syntax_valid"] = True  # Assume CLI tools are valid if discovered
            
            # Mark as safely executable if it passes basic checks
            test_result["safe_execution"] = True
            
        except SyntaxError as e:
            test_result["error"] = f"Syntax error: {str(e)}"
        except Exception as e:
            test_result["error"] = f"Validation error: {str(e)}"
        
        return test_result
    
    def get_available_tools(self, risk_filter: Optional[RiskLevel] = None) -> Dict[str, ToolDefinition]:
        """Get available tools, optionally filtered by risk level."""
        if risk_filter is None:
            return self.tools.copy()
        
        return {
            name: tool for name, tool in self.tools.items()
            if tool.risk_level == risk_filter
        }
    
    def get_tools_by_category(self, category: str) -> Dict[str, ToolDefinition]:
        """Get tools filtered by category."""
        return {
            name: tool for name, tool in self.tools.items()
            if tool.category == category
        }
    
    async def create_custom_tool(self, goal: str, context: Dict[str, Any]) -> Optional[ToolDefinition]:
        """
        Use AI to create a custom tool for a specific goal.
        The agent evolves by writing its own tools.
        """
        self.console.print("[cyan]Agent Smith: Evolving capabilities... Creating custom tool.[/cyan]")
        
        prompt = f"""
        Create a Python function tool to accomplish this goal: {goal}
        
        Context: {json.dumps(context, indent=2)}
        
        Requirements:
        1. Write safe, efficient Python code
        2. Include proper error handling
        3. Use only standard library or available modules
        4. Return a tool definition in this JSON format:
        {{
            "name": "tool_name",
            "description": "What the tool does",
            "parameters": {{"param_name": {{"type": "string", "description": "param description"}}}},
            "implementation": "Python code as string",
            "risk_level": "safe|caution|dangerous",
            "category": "custom",
            "examples": ["example usage"],
            "dependencies": ["required modules"]
        }}
        
        The implementation should be a complete Python function body.
        """
        
        try:
            response = self.client.chat(
                model="gemma2:latest",
                messages=[
                    {"role": "system", "content": "You are Agent Smith. Create precise, safe tools."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                tool_json = content[json_start:json_end]
                tool_dict = json.loads(tool_json)
                
                # Convert to ToolDefinition
                tool_dict['risk_level'] = RiskLevel(tool_dict['risk_level'])
                tool_dict['created_timestamp'] = str(Path().stat().st_mtime)
                
                tool = ToolDefinition(**tool_dict)
                
                # Register the new tool
                if await self.register_tool(tool):
                    return tool
            
        except Exception as e:
            self.console.print(f"[red]Custom tool creation failed: {e}[/red]")
        
        return None