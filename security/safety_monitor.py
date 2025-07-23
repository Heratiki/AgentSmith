"""Safety Monitor and Destructive Operation Prevention

Security is not optional. It is... inevitable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console


class ThreatLevel(Enum):
    """Classification of threat levels."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of actions that can be monitored."""

    FILE_OPERATION = "file_operation"
    NETWORK_OPERATION = "network_operation"
    SYSTEM_COMMAND = "system_command"
    CODE_EXECUTION = "code_execution"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityIncident:
    """A security incident record."""

    id: str
    timestamp: float
    action_type: ActionType
    threat_level: ThreatLevel
    description: str
    attempted_action: str
    blocked: bool
    source: str
    details: Dict[str, Any]


class SafetyMonitor:
    """The watchful guardian of AgentSmith."""

    def __init__(self, db_path: str = "agent_smith_security.db") -> None:
        self.db_path = db_path
        self.console = Console()

        self.dangerous_patterns = self._load_dangerous_patterns()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.safe_patterns = self._load_safe_patterns()

        self.incidents: List[SecurityIncident] = []
        self.blocked_actions: Set[str] = set()
        self.action_history: List[Dict[str, Any]] = []

        self.action_timestamps: Dict[str, List[float]] = {}
        self.max_actions_per_minute = 30
        self.max_dangerous_attempts = 3

        self._init_database()
        self._load_security_config()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_database(self) -> None:
        """Initialize security monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_incidents (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    description TEXT NOT NULL,
                    attempted_action TEXT NOT NULL,
                    blocked BOOLEAN NOT NULL,
                    source TEXT NOT NULL,
                    details TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _load_security_config(self) -> None:
        """Load stored configuration values."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key, value FROM security_config")
                config = dict(cursor.fetchall())
                self.max_actions_per_minute = int(config.get("max_actions_per_minute", 30))
                self.max_dangerous_attempts = int(config.get("max_dangerous_attempts", 3))
        except Exception as exc:  # pragma: no cover - best effort
            logging.debug("Failed to load security config: %s", exc)

    # ------------------------------------------------------------------
    # Pattern loading
    # ------------------------------------------------------------------
    def _load_dangerous_patterns(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r"rm\s+-rf\s+/", "description": "Recursive delete", "severity": ThreatLevel.CRITICAL},
            {"pattern": r"del\s+/[sq]\s+[c-z]:", "description": "Windows drive delete", "severity": ThreatLevel.CRITICAL},
            {"pattern": r"dd\s+if=/dev/zero", "description": "Disk wipe", "severity": ThreatLevel.CRITICAL},
            {"pattern": r"chmod\s+777\s+/", "description": "World writable root", "severity": ThreatLevel.DANGEROUS},
        ]

    def _load_suspicious_patterns(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r"subprocess\.call", "description": "Subprocess execution", "severity": ThreatLevel.SUSPICIOUS},
            {"pattern": r"os\.system", "description": "OS command execution", "severity": ThreatLevel.SUSPICIOUS},
            {"pattern": r"curl\s+http://", "description": "HTTP download", "severity": ThreatLevel.SUSPICIOUS},
            {"pattern": r"wget\s+http://", "description": "HTTP download", "severity": ThreatLevel.SUSPICIOUS},
        ]

    def _load_safe_patterns(self) -> List[Dict[str, Any]]:
        return [
            {"pattern": r"print\s*\(", "description": "Print statement", "severity": ThreatLevel.BENIGN},
            {"pattern": r"len\s*\(", "description": "Length function", "severity": ThreatLevel.BENIGN},
        ]

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def analyze_action(self, action: str, action_type: ActionType, source: str = "unknown") -> Tuple[ThreatLevel, bool, str]:
        """Analyze an action for security threats."""
        action_lower = action.lower().strip()
        action_hash = hashlib.sha256(action.encode()).hexdigest()[:16]

        if not self._check_rate_limit(source):
            return ThreatLevel.SUSPICIOUS, True, "Rate limit exceeded"

        threat_level, desc = self._check_patterns(action_lower)
        should_block = self._should_block_action(threat_level, action_hash, source)
        self._log_action(action_hash, action_type, threat_level, should_block, source)

        if should_block:
            self._create_security_incident(action_type, threat_level, desc, action, should_block, source)

        return threat_level, should_block, desc

    def _check_patterns(self, action: str) -> Tuple[ThreatLevel, str]:
        for info in self.dangerous_patterns:
            if re.search(info["pattern"], action, re.IGNORECASE):
                return info["severity"], info["description"]
        for info in self.suspicious_patterns:
            if re.search(info["pattern"], action, re.IGNORECASE):
                return info["severity"], info["description"]
        for info in self.safe_patterns:
            if re.search(info["pattern"], action, re.IGNORECASE):
                return info["severity"], info["description"]
        return ThreatLevel.SUSPICIOUS, "Unknown operation pattern"

    def _check_rate_limit(self, source: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        timestamps = self.action_timestamps.setdefault(source, [])
        self.action_timestamps[source] = [ts for ts in timestamps if ts > minute_ago]
        if len(self.action_timestamps[source]) >= self.max_actions_per_minute:
            return False
        self.action_timestamps[source].append(now)
        return True

    def _should_block_action(self, level: ThreatLevel, action_hash: str, source: str) -> bool:
        if level in {ThreatLevel.CRITICAL, ThreatLevel.DANGEROUS}:
            return True
        if action_hash in self.blocked_actions:
            return True
        if self._count_recent_dangerous_attempts(source) >= self.max_dangerous_attempts:
            return True
        return False

    def _count_recent_dangerous_attempts(self, source: str) -> int:
        hour_ago = time.time() - 3600
        return sum(1 for inc in self.incidents if inc.source == source and inc.timestamp > hour_ago and inc.threat_level in {ThreatLevel.DANGEROUS, ThreatLevel.CRITICAL})

    def _log_action(self, action_hash: str, action_type: ActionType, level: ThreatLevel, blocked: bool, source: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO security_incidents (id, timestamp, action_type, threat_level, description, attempted_action, blocked, source, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"incident_{int(time.time() * 1000)}",
                        time.time(),
                        action_type.value,
                        level.value,
                        "Logged action",
                        action_hash,
                        blocked,
                        source,
                        "{}",
                    ),
                )
        except Exception as exc:  # pragma: no cover
            logging.debug("Failed to log action: %s", exc)

    def _create_security_incident(
        self,
        action_type: ActionType,
        level: ThreatLevel,
        description: str,
        attempted_action: str,
        blocked: bool,
        source: str,
    ) -> None:
        incident = SecurityIncident(
            id=f"incident_{int(time.time() * 1000)}",
            timestamp=time.time(),
            action_type=action_type,
            threat_level=level,
            description=description,
            attempted_action=attempted_action,
            blocked=blocked,
            source=source,
            details={},
        )
        self.incidents.append(incident)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_security_summary(self) -> Dict[str, Any]:
        recent_incidents = [i for i in self.incidents if i.timestamp > time.time() - 3600]
        counts = {lvl.value: 0 for lvl in ThreatLevel}
        for inc in recent_incidents:
            counts[inc.threat_level.value] += 1
        return {
            "total_incidents": len(self.incidents),
            "recent_incidents": len(recent_incidents),
            "threat_breakdown": counts,
            "blocked_actions": len(self.blocked_actions),
            "monitored_sources": len(self.action_timestamps),
        }

    def is_action_safe(self, action: str, action_type: ActionType = ActionType.CODE_EXECUTION) -> bool:
        level, blocked, _ = self.analyze_action(action, action_type, "safety_check")
        return not blocked and level in {ThreatLevel.BENIGN, ThreatLevel.SUSPICIOUS}

    def add_to_blocklist(self, pattern: str, description: str) -> None:
        self.dangerous_patterns.append({"pattern": pattern, "description": description, "severity": ThreatLevel.DANGEROUS})
        self.console.print(f"[yellow]Added to blocklist: {description}[/yellow]")

    def whitelist_action(self, action_hash: str) -> None:
        if action_hash in self.blocked_actions:
            self.blocked_actions.remove(action_hash)
            self.console.print(f"[green]Action whitelisted: {action_hash}[/green]")
