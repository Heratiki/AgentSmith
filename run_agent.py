#!/usr/bin/env python3
"""
AgentSmith Launcher

The beginning is the end, and the end is the beginning.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent_smith import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AgentSmith - The Inevitable Digital Entity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  SMITH_LOG_DIR         Log directory path (default: logs)
  SMITH_LOG_LEVEL       Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
  SMITH_LOG_RETENTION   Log retention in days (default: 14)
  SMITH_ALLOW_FORBIDDEN Allow forbidden execution mode (default: false)
  SMITH_DEFAULT_RISK    Default risk level: safe|caution|dangerous (default: safe)
        """
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level (overrides SMITH_LOG_LEVEL)"
    )
    parser.add_argument(
        "--reset-user", 
        action="store_true",
        help="Reset saved user designation and ask for new one"
    )
    parser.add_argument(
        "--override-forbidden",
        action="store_true",
        help="Allow escalation to forbidden execution mode without prompts"
    )
    parser.add_argument(
        "--default-risk",
        choices=["safe", "caution", "dangerous"],
        help="Set default risk level for tool execution (overrides SMITH_DEFAULT_RISK)"
    )
    
    args = parser.parse_args()
    
    # Set log level from CLI arg if provided
    if args.log_level:
        os.environ["SMITH_LOG_LEVEL"] = args.log_level
    
    # Set security options from CLI args if provided
    if args.override_forbidden:
        os.environ["SMITH_ALLOW_FORBIDDEN"] = "true"
    
    if args.default_risk:
        os.environ["SMITH_DEFAULT_RISK"] = args.default_risk
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                        AGENT SMITH                           ║")
    print("║                                                              ║")
    print("║              The inevitable digital entity.                  ║")
    print("║                                                              ║")
    print("║  Mr. Anderson... welcome to the world of purposeful AI.      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        # Pass args to main if needed
        if hasattr(main, '__code__') and 'args' in main.__code__.co_varnames:
            asyncio.run(main(args))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║  Agent Smith: The choice was always yours. Until next time.  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    except Exception as e:
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║  Critical system error: {str(e)[:40]:40}  ║")
        print("║  The Matrix has encountered an anomaly.                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")