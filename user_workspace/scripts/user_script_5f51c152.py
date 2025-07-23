
# Sandboxed tool execution
from pathlib import Path
import json

# Injected parameters
dirpath = '.'

# Create execution result storage
execution_result = {"success": True, "details": [], "files_created": [], "errors": []}

try:
    # Original tool implementation
    list(Path(dirpath).iterdir())
    
except Exception as e:
    execution_result["success"] = False
    execution_result["errors"].append(str(e))

# Output results as JSON for parsing
print(json.dumps(execution_result))
