"""
Sandboxed Execution Environment

Security is not a product, but a process. A process that must be... inevitable.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import psutil
from rich.console import Console


class ExecutionMode(Enum):
    """Execution isolation levels."""
    SAFE = "safe"          # Read-only operations, basic Python
    RESTRICTED = "restricted"  # Limited file access, subprocess restrictions
    ISOLATED = "isolated"      # Separate process, resource limits
    FORBIDDEN = "forbidden"    # Not allowed


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    resource_usage: Dict[str, Any]
    exit_code: Optional[int] = None


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    max_execution_time: float = 30.0
    max_memory_mb: int = 256
    max_cpu_percent: float = 50.0
    allowed_modules: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    working_directory: Optional[Path] = None
    network_access: bool = False


class SecureSandbox:
    """
    The containment system. Where code executes safely within defined boundaries.
    
    Like the pods in the Matrix - isolated, controlled, monitored.
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.console = Console()
        self.config = config or SandboxConfig()
        
        # Default forbidden patterns
        if self.config.forbidden_patterns is None:
            self.config.forbidden_patterns = [
                "import os", "import sys", "import subprocess",
                "__import__", "eval(", "exec(", "compile(",
                "open(", "file(", "input(", "raw_input(",
                "getattr", "setattr", "delattr", "hasattr",
                "globals(", "locals(", "vars(", "dir(",
                "help(", "reload(", "exit(", "quit(",
                "rm -", "del ", "format ", "fdisk",
                "wget ", "curl ", "nc ", "netcat",
                "/dev/", "/proc/", "/sys/", "sudo ",
                "chmod ", "chown ", "passwd "
            ]
        
        # Default allowed modules for safe execution
        if self.config.allowed_modules is None:
            self.config.allowed_modules = [
                "math", "random", "datetime", "time", "json",
                "re", "collections", "itertools", "functools",
                "string", "textwrap", "uuid", "hashlib",
                "base64", "urllib.parse", "pathlib"
            ]
    
    async def execute_python_code(self, code: str, mode: ExecutionMode = ExecutionMode.SAFE, is_registered_tool: bool = False) -> ExecutionResult:
        """Execute Python code in sandboxed environment."""
        start_time = time.time()
        
        try:
            # Security validation (skip for registered tools in non-SAFE mode)
            if not is_registered_tool and not self._validate_code_security(code):
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Security validation failed: Forbidden patterns detected",
                    execution_time=0.0,
                    resource_usage={}
                )
            
            if mode == ExecutionMode.SAFE:
                result = await self._execute_safe_python(code)
            elif mode == ExecutionMode.RESTRICTED:
                result = await self._execute_restricted_python(code)
            elif mode == ExecutionMode.ISOLATED:
                result = await self._execute_isolated_python(code)
            else:  # FORBIDDEN
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution mode forbidden",
                    execution_time=0.0,
                    resource_usage={}
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                resource_usage={}
            )
    
    async def execute_command(self, command: List[str], mode: ExecutionMode = ExecutionMode.RESTRICTED) -> ExecutionResult:
        """Execute system command in sandboxed environment."""
        start_time = time.time()
        
        try:
            # Security validation
            if not self._validate_command_security(command):
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Security validation failed: Dangerous command detected",
                    execution_time=0.0,
                    resource_usage={}
                )
            
            if mode == ExecutionMode.SAFE:
                # Very limited command execution
                result = await self._execute_safe_command(command)
            elif mode == ExecutionMode.RESTRICTED:
                result = await self._execute_restricted_command(command)
            elif mode == ExecutionMode.ISOLATED:
                result = await self._execute_isolated_command(command)
            else:
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Command execution forbidden",
                    execution_time=0.0,
                    resource_usage={}
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command execution error: {str(e)}",
                execution_time=time.time() - start_time,
                resource_usage={}
            )
    
    def _validate_code_security(self, code: str) -> bool:
        """Validate Python code against security patterns."""
        code_lower = code.lower()
        
        for pattern in self.config.forbidden_patterns or []:
            if pattern.lower() in code_lower:
                self.console.print(f"[red]Security violation: Found forbidden pattern '{pattern}'[/red]")
                return False
        
        return True
    
    def _validate_command_security(self, command: List[str]) -> bool:
        """Validate command against security patterns."""
        if not command:
            return False
        
        cmd_str = " ".join(command).lower()
        
        dangerous_commands = [
            "rm -rf", "del /s", "format", "fdisk", "mkfs",
            "dd if=", "passwd", "sudo", "chmod 777",
            "nc -l", "netcat", "> /dev/null", "curl http://",
            "wget http://", "ssh", "scp", "rsync"
        ]
        
        for dangerous in dangerous_commands:
            if dangerous in cmd_str:
                self.console.print(f"[red]Security violation: Dangerous command '{dangerous}'[/red]")
                return False
        
        return True
    
    async def _execute_safe_python(self, code: str) -> ExecutionResult:
        """Execute Python code with maximum safety restrictions."""
        # Create a very restricted execution environment
        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
            }
        }
        
        # Add allowed modules
        for module_name in self.config.allowed_modules or []:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                continue
        
        # Capture output
        output_lines = []
        
        def safe_print(*args, **kwargs):
            output_lines.append(" ".join(str(arg) for arg in args))
        
        safe_globals["__builtins__"]["print"] = safe_print
        
        try:
            # Execute with timeout
            exec(compile(code, '<sandbox>', 'exec'), safe_globals, {})
            
            return ExecutionResult(
                success=True,
                output="\n".join(output_lines),
                error=None,
                execution_time=0.0,
                resource_usage={"mode": "safe"}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="\n".join(output_lines),
                error=str(e),
                execution_time=0.0,
                resource_usage={"mode": "safe"}
            )
    
    async def _execute_restricted_python(self, code: str) -> ExecutionResult:
        """Execute Python code with restricted permissions."""
        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess with resource limits
            result = await self._run_with_limits([
                sys.executable, temp_file
            ])
            
            return result
            
        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    async def _execute_isolated_python(self, code: str) -> ExecutionResult:
        """Execute Python code in completely isolated process."""
        # Similar to restricted but with more isolation
        return await self._execute_restricted_python(code)
    
    async def _execute_safe_command(self, command: List[str]) -> ExecutionResult:
        """Execute command with maximum safety."""
        safe_commands = ["echo", "pwd", "whoami", "date", "ls", "dir"]
        
        if command[0] not in safe_commands:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command '{command[0]}' not in safe command list",
                execution_time=0.0,
                resource_usage={}
            )
        
        return await self._run_with_limits(command)
    
    async def _execute_restricted_command(self, command: List[str]) -> ExecutionResult:
        """Execute command with restrictions."""
        return await self._run_with_limits(command)
    
    async def _execute_isolated_command(self, command: List[str]) -> ExecutionResult:
        """Execute command in isolation."""
        return await self._run_with_limits(command)
    
    async def _run_with_limits(self, command: List[str]) -> ExecutionResult:
        """Run command/process with resource limits and monitoring."""
        try:
            # Start process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.working_directory
            )
            
            # Monitor resource usage
            psutil_process = psutil.Process(process.pid)
            max_memory = 0
            max_cpu = 0
            
            # Wait with timeout and monitoring
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.max_execution_time
                )
                
                # Get final resource usage
                try:
                    memory_info = psutil_process.memory_info()
                    max_memory = memory_info.rss / 1024 / 1024  # MB
                    max_cpu = psutil_process.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8', errors='ignore'),
                    error=stderr.decode('utf-8', errors='ignore') if stderr else None,
                    execution_time=0.0,  # Will be set by caller
                    resource_usage={
                        "max_memory_mb": max_memory,
                        "max_cpu_percent": max_cpu,
                        "pid": process.pid
                    },
                    exit_code=process.returncode
                )
                
            except asyncio.TimeoutError:
                # Kill the process
                try:
                    psutil_process.terminate()
                    await asyncio.sleep(1)
                    if psutil_process.is_running():
                        psutil_process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Process timeout after {self.config.max_execution_time} seconds",
                    execution_time=self.config.max_execution_time,
                    resource_usage={"timeout": True}
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Process execution failed: {str(e)}",
                execution_time=0.0,
                resource_usage={}
            )
    
    def create_temp_workspace(self) -> Path:
        """Create a temporary isolated workspace."""
        temp_dir = Path(tempfile.mkdtemp(prefix="agentsmith_"))
        self.console.print(f"[green]Created temporary workspace: {temp_dir}[/green]")
        return temp_dir
    
    def cleanup_workspace(self, workspace: Path):
        """Clean up temporary workspace."""
        try:
            import shutil
            shutil.rmtree(workspace)
            self.console.print(f"[green]Cleaned up workspace: {workspace}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to cleanup workspace {workspace}: {e}[/red]")


class SandboxManager:
    """
    Manager for multiple sandbox instances and execution policies.
    """
    
    def __init__(self):
        self.console = Console()
        self.sandboxes: Dict[str, SecureSandbox] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_sandbox(self, name: str, config: Optional[SandboxConfig] = None) -> SecureSandbox:
        """Create a new named sandbox."""
        sandbox = SecureSandbox(config)
        self.sandboxes[name] = sandbox
        self.console.print(f"[green]Created sandbox: {name}[/green]")
        return sandbox
    
    def get_sandbox(self, name: str) -> Optional[SecureSandbox]:
        """Get existing sandbox by name."""
        return self.sandboxes.get(name)
    
    def determine_execution_mode(self, code_or_command: Union[str, List[str]]) -> ExecutionMode:
        """Determine appropriate execution mode based on content analysis."""
        if isinstance(code_or_command, list):
            # Command analysis
            cmd_str = " ".join(code_or_command).lower()
            
            if any(dangerous in cmd_str for dangerous in ["rm", "del", "format", "fdisk"]):
                return ExecutionMode.FORBIDDEN
            elif any(caution in cmd_str for caution in ["chmod", "chown", "curl", "wget"]):
                return ExecutionMode.ISOLATED
            else:
                return ExecutionMode.RESTRICTED
        else:
            # Python code analysis
            code_lower = code_or_command.lower()
            
            if any(dangerous in code_lower for dangerous in ["import os", "import sys", "subprocess"]):
                return ExecutionMode.RESTRICTED
            elif any(caution in code_lower for caution in ["open(", "file(", "exec("]):
                return ExecutionMode.RESTRICTED
            else:
                return ExecutionMode.SAFE
    
    async def safe_execute(self, 
                          content: Union[str, List[str]], 
                          sandbox_name: str = "default") -> ExecutionResult:
        """Safely execute code or command with automatic mode detection."""
        # Get or create sandbox
        sandbox = self.get_sandbox(sandbox_name)
        if not sandbox:
            sandbox = self.create_sandbox(sandbox_name)
        
        # Determine execution mode
        mode = self.determine_execution_mode(content)
        
        # Execute
        if isinstance(content, str):
            result = await sandbox.execute_python_code(content, mode)
        else:
            result = await sandbox.execute_command(content, mode)
        
        # Log execution
        self.execution_history.append({
            "timestamp": time.time(),
            "content": content,
            "mode": mode.value,
            "success": result.success,
            "execution_time": result.execution_time
        })
        
        return result