#!/usr/bin/env python3
"""
Test sandboxed execution
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_smith import AgentSmith

async def test_sandbox():
    agent = AgentSmith()
    
    print("Testing sandboxed file creation...")
    
    try:
        final_state = await agent.run("Create a file called test_sandbox.txt with the content 'Hello from sandbox!'", "TestUser")
        
        execution_results = final_state.get('execution_results', [])
        if execution_results:
            result = execution_results[0]
            print(f"\nExecution successful: {result.get('success')}")
            print(f"Resource usage: {result.get('resource_usage', {})}")
            print(f"Execution time: {result.get('execution_time', 'unknown')}")
            
            # Check if file was actually created
            test_file = Path("test_sandbox.txt")
            if test_file.exists():
                content = test_file.read_text()
                print(f"File created successfully with content: '{content}'")
                test_file.unlink()  # Clean up
            else:
                print("File was not created")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sandbox())