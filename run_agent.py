#!/usr/bin/env python3
"""
AgentSmith Launcher

The beginning is the end, and the end is the beginning.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_smith import main

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                        AGENT SMITH                          ║")
    print("║                                                              ║")
    print("║              The inevitable digital entity.                 ║")
    print("║                                                              ║")
    print("║  Mr. Anderson... welcome to the world of purposeful AI.     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  Agent Smith: The choice was always yours. Until next time. ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    except Exception as e:
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  Critical system error: {str(e)[:40]:40}  ║")
        print("║  The Matrix has encountered an anomaly.                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")