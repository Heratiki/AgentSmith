#!/usr/bin/env python3
"""
Test the new permission-based security system
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_smith import AgentSmith

async def test_permission():
    agent = AgentSmith()
    
    print("Testing permission-based security system")
    print("="*60)
    print("This should now ask for permission instead of blocking...")
    
    # Test weather request that should ask for permission
    final_state = await agent.run("Search for weather information about Miami, FL", "TestUser")

if __name__ == "__main__":
    asyncio.run(test_permission())