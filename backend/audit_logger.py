"""
CyberGaze - Audit Logger
Structured event logging with brute-force detection and user lockout.
Append-only JSON log file for tamper evidence.
"""

import json
import os
import time
from datetime import datetime
from collections import defaultdict


# Log file path
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'cybergaze.audit.log')

# Brute-force thresholds
MAX_FAILURES = 3          # Max failures before lockout
FAILURE_WINDOW = 300      # 5-minute window for counting failures
LOCKOUT_DURATION = 600    # 10-minute lockout duration


class AuditLogger:
    """
    Append-only audit logger with structured JSON events.
    Tracks authentication attempts and detects brute-force attacks.
    """

    # Event types
    EVENT_ENROLL = 'ENROLL'
    EVENT_VERIFY = 'VERIFY'
    EVENT_LOCK = 'LOCK'
    EVENT_UNLOCK = 'UNLOCK'
    EVENT_SPOOF = 'SPOOF_ATTEMPT'
    EVENT_LIVENESS_FAIL = 'LIVENESS_FAIL'
    EVENT_LOCKOUT = 'LOCKOUT'
    EVENT_UNLOCK_ACCOUNT = 'ACCOUNT_UNLOCKED'

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        # In-memory tracking for brute-force detection
        self._failure_tracker = defaultdict(list)
        self._lockouts = {}  # user_id -> lockout_expiry_timestamp
        print(f"Audit logger initialized. Log file: {LOG_FILE}")

    def log_event(self, event_type: str, user_id: str = None,
                  folder_id: str = None, success: bool = True,
                  details: dict = None):
        """
        Log a security event to the audit log.
        
        Args:
            event_type: One of the EVENT_* constants
            user_id: User associated with the event
            folder_id: Folder associated with the event
            success: Whether the action succeeded
            details: Additional event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'epoch': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'folder_id': folder_id,
            'success': success,
            'details': details or {}
        }

        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"Audit log write error: {e}")

        # Track failures for brute-force detection
        if not success and event_type in (self.EVENT_VERIFY, self.EVENT_SPOOF, self.EVENT_LIVENESS_FAIL):
            self._track_failure(user_id)

    def _track_failure(self, user_id: str):
        """Track failed authentication attempt for brute-force detection."""
        if not user_id:
            return

        now = time.time()
        self._failure_tracker[user_id].append(now)

        # Clean old entries outside the window
        cutoff = now - FAILURE_WINDOW
        self._failure_tracker[user_id] = [
            t for t in self._failure_tracker[user_id] if t > cutoff
        ]

        # Check if threshold exceeded
        if len(self._failure_tracker[user_id]) >= MAX_FAILURES:
            self._lockout_user(user_id)

    def _lockout_user(self, user_id: str):
        """Lock out a user due to too many failed attempts."""
        lockout_until = time.time() + LOCKOUT_DURATION
        self._lockouts[user_id] = lockout_until

        self.log_event(
            self.EVENT_LOCKOUT,
            user_id=user_id,
            success=True,
            details={
                'reason': 'Too many failed verification attempts',
                'failures': len(self._failure_tracker[user_id]),
                'lockout_duration_seconds': LOCKOUT_DURATION
            }
        )

        # Update database status
        try:
            from database import update_user_status
            update_user_status(user_id, 'locked')
        except Exception as e:
            print(f"Failed to update user lockout status: {e}")

        # Clear failure tracker for this user
        self._failure_tracker[user_id] = []
        print(f"SECURITY: User '{user_id}' locked out for {LOCKOUT_DURATION}s due to brute-force")

    def is_locked_out(self, user_id: str) -> dict:
        """
        Check if a user is currently locked out.
        
        Returns:
            dict with locked status and remaining time
        """
        if user_id in self._lockouts:
            remaining = self._lockouts[user_id] - time.time()
            if remaining > 0:
                return {
                    'locked': True,
                    'remaining_seconds': int(remaining),
                    'message': f'Account locked. Try again in {int(remaining)} seconds.'
                }
            else:
                # Lockout expired — remove and unlock
                del self._lockouts[user_id]
                try:
                    from database import update_user_status
                    update_user_status(user_id, 'active')
                except Exception:
                    pass

        return {'locked': False}

    def analyze_threats(self, user_id: str = None) -> dict:
        """
        Analyze recent threats for a user or system-wide.
        
        Returns:
            dict with threat analysis summary
        """
        now = time.time()
        cutoff = now - FAILURE_WINDOW

        if user_id:
            recent_failures = [
                t for t in self._failure_tracker.get(user_id, []) if t > cutoff
            ]
            return {
                'user_id': user_id,
                'recent_failures': len(recent_failures),
                'max_allowed': MAX_FAILURES,
                'is_locked': user_id in self._lockouts and self._lockouts[user_id] > now,
                'threat_level': 'high' if len(recent_failures) >= MAX_FAILURES - 1 else
                               'medium' if len(recent_failures) >= 1 else 'low'
            }

        # System-wide analysis
        all_failures = sum(
            len([t for t in times if t > cutoff])
            for times in self._failure_tracker.values()
        )
        active_lockouts = sum(
            1 for exp in self._lockouts.values() if exp > now
        )

        return {
            'total_recent_failures': all_failures,
            'active_lockouts': active_lockouts,
            'monitored_users': len(self._failure_tracker),
            'threat_level': 'high' if active_lockouts > 0 else
                           'medium' if all_failures >= 2 else 'low'
        }

    def get_audit_logs(self, limit: int = 50, offset: int = 0,
                       event_type: str = None, user_id: str = None) -> dict:
        """
        Retrieve audit log entries with optional filtering.
        
        Args:
            limit: Max number of entries to return
            offset: Number of entries to skip
            event_type: Filter by event type
            user_id: Filter by user ID
            
        Returns:
            dict with log entries and pagination info
        """
        entries = []

        try:
            if not os.path.exists(LOG_FILE):
                return {'entries': [], 'total': 0}

            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            # Parse and filter
            for line in reversed(all_lines):  # Most recent first
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)

                    if event_type and entry.get('event_type') != event_type:
                        continue
                    if user_id and entry.get('user_id') != user_id:
                        continue

                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

            total = len(entries)
            entries = entries[offset:offset + limit]

            return {
                'entries': entries,
                'total': total,
                'limit': limit,
                'offset': offset
            }

        except Exception as e:
            print(f"Error reading audit logs: {e}")
            return {'entries': [], 'total': 0, 'error': str(e)}


# Singleton instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get singleton AuditLogger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
