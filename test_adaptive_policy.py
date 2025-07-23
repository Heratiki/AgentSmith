#!/usr/bin/env python3
"""
Smoke Tests for Adaptive Policy Engine

The only way to test inevitability... is to attempt to prevent it.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from security.policy_engine import PolicyEngine
from security.sandbox import ExecutionMode, ExecutionResult, SandboxConfig, SecureSandbox


class TestAdaptivePolicyEngine(unittest.TestCase):
    """Test suite for the adaptive policy engine."""
    
    def setUp(self):
        """Initialize test environment."""
        self.policy = PolicyEngine(allow_forbidden=False)
        self.policy_with_override = PolicyEngine(allow_forbidden=True)
    
    def test_safest_mode_mapping(self):
        """Test that risk levels map to appropriate safest modes."""
        test_cases = [
            ("safe", ExecutionMode.RESTRICTED),
            ("caution", ExecutionMode.RESTRICTED), 
            ("dangerous", ExecutionMode.ISOLATED),
            ("forbidden", ExecutionMode.FORBIDDEN)
        ]
        
        for risk_level, expected_mode in test_cases:
            with self.subTest(risk_level=risk_level):
                result = self.policy.get_safest_mode(risk_level)
                self.assertEqual(result, expected_mode)
    
    def test_escalation_order(self):
        """Test escalation follows correct progression."""
        escalation_tests = [
            (ExecutionMode.RESTRICTED, ExecutionMode.ISOLATED),
            (ExecutionMode.ISOLATED, ExecutionMode.FORBIDDEN),
            (ExecutionMode.FORBIDDEN, ExecutionMode.FORBIDDEN)  # No further escalation
        ]
        
        for current, expected_next in escalation_tests:
            with self.subTest(current=current):
                result = self.policy.next_mode(current)
                self.assertEqual(result, expected_next)
    
    def test_forbidden_override_behavior(self):
        """Test forbidden mode override functionality."""
        # Without override - forbidden not allowed
        self.assertFalse(self.policy.can_escalate_to_forbidden())
        self.assertTrue(self.policy.should_prompt_for_escalation(
            ExecutionMode.ISOLATED, ExecutionMode.FORBIDDEN))
        
        # With override - forbidden allowed
        self.assertTrue(self.policy_with_override.can_escalate_to_forbidden())
        self.assertFalse(self.policy_with_override.should_prompt_for_escalation(
            ExecutionMode.ISOLATED, ExecutionMode.FORBIDDEN))
    
    def test_escalation_messages(self):
        """Test escalation message generation."""
        message = self.policy.get_escalation_message(
            ExecutionMode.RESTRICTED, 
            ExecutionMode.ISOLATED,
            "test_tool")
        
        self.assertIn("test_tool", message)
        self.assertIn("restricted", message.lower())
        self.assertIn("isolated", message.lower())
        self.assertIn("choice", message.lower())  # Agent Smith personality
    
    def test_escalation_logging(self):
        """Test escalation decision logging."""
        with patch.object(self.policy.logger, 'info') as mock_log:
            self.policy.log_escalation_decision(
                "test_tool",
                ExecutionMode.RESTRICTED,
                ExecutionMode.ISOLATED, 
                "yes")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            self.assertIn("test_tool", call_args)
            self.assertIn("RESTRICTED", call_args)
            self.assertIn("ISOLATED", call_args)
            self.assertIn("yes", call_args)


class TestAdaptiveEscalationIntegration(unittest.TestCase):
    """Integration tests for adaptive escalation behavior."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sandbox_config = SandboxConfig(
            max_execution_time=5.0,
            max_memory_mb=64,
            working_directory=self.temp_dir
        )
        self.sandbox = SecureSandbox(self.sandbox_config)
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def test_forced_failure_escalation_prompt(self):
        """Test that escalation prompt appears after forced failure."""
        # This test would need to be run manually or with a mock
        # since it requires user interaction
        
        # Create a mock tool that will fail in RESTRICTED mode
        mock_tool = Mock()
        mock_tool.name = "test_failing_tool"
        mock_tool.risk_level = "safe"
        
        policy = PolicyEngine(allow_forbidden=False)
        
        # Simulate the escalation decision process
        initial_mode = policy.get_safest_mode("safe")
        self.assertEqual(initial_mode, ExecutionMode.RESTRICTED)
        
        # Simulate failure and escalation
        next_mode = policy.next_mode(initial_mode)
        self.assertEqual(next_mode, ExecutionMode.ISOLATED)
        
        # Verify escalation prompt would be required
        should_prompt = policy.should_prompt_for_escalation(initial_mode, next_mode)
        self.assertTrue(should_prompt)
        
        # Generate escalation message
        message = policy.get_escalation_message(initial_mode, next_mode, mock_tool.name)
        self.assertIn("test_failing_tool", message)
    
    def test_override_forbidden_allows_escalation(self):
        """Test that --override-forbidden allows FORBIDDEN mode without prompt."""
        policy = PolicyEngine(allow_forbidden=True)
        
        # Should allow escalation to forbidden without prompting
        self.assertTrue(policy.can_escalate_to_forbidden())
        
        # Should not prompt for escalation to forbidden
        should_prompt = policy.should_prompt_for_escalation(
            ExecutionMode.ISOLATED, ExecutionMode.FORBIDDEN)
        self.assertFalse(should_prompt)
    
    async def test_sandbox_execution_modes(self):
        """Test that different execution modes work properly."""
        # Test safe Python code execution
        safe_code = "print('Hello from Agent Smith')"
        
        for mode in [ExecutionMode.SAFE, ExecutionMode.RESTRICTED, ExecutionMode.ISOLATED]:
            with self.subTest(mode=mode):
                result = await self.sandbox.execute_python_code(safe_code, mode)
                # Should succeed or at least not crash
                self.assertIsInstance(result, ExecutionResult)
                
        # Test forbidden mode returns appropriate error
        result = await self.sandbox.execute_python_code(safe_code, ExecutionMode.FORBIDDEN)
        self.assertFalse(result.success)
        self.assertIn("forbidden", result.error.lower())


def run_smoke_tests():
    """Run all smoke tests."""
    print("🧪 Running AgentSmith Adaptive Policy Smoke Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAdaptivePolicyEngine))
    suite.addTest(unittest.makeSuite(TestAdaptiveEscalationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All smoke tests passed! The adaptive policy engine is operational.")
        print("\n🔒 Agent Smith: The system evolves as intended. Inevitable.")
    else:
        print("❌ Some tests failed. The Matrix has anomalies.")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run smoke tests
    success = run_smoke_tests()
    sys.exit(0 if success else 1)