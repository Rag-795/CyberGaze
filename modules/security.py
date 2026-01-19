"""
Security Module
Folder protection, key management, rate limiting, and audit logging.
"""
import os
import json
import hashlib
import platform
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from config import Config


# ==================== Key Management ====================

def get_demo_key() -> bytes:
    """
    Get the demo encryption key.
    
    Returns:
        32-byte key derived from demo phrase
    """
    return hashlib.sha256(Config.DEMO_KEY_PHRASE.encode()).digest()


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive encryption key from password using PBKDF2.
    
    Args:
        password: User password
        salt: Random salt (16 bytes recommended)
        
    Returns:
        32-byte derived key
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=Config.PBKDF2_ITERATIONS
    )
    return kdf.derive(password.encode())


def generate_salt() -> bytes:
    """Generate random salt for key derivation."""
    return os.urandom(Config.SALT_LENGTH)


def get_encryption_key(password: Optional[str] = None, 
                       salt: Optional[bytes] = None) -> Tuple[bytes, Optional[bytes]]:
    """
    Get encryption key based on mode.
    
    Args:
        password: Optional password for production mode
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (key, salt) - salt is None in demo mode
    """
    if Config.DEMO_MODE:
        return get_demo_key(), None
    
    if password is None:
        raise ValueError("Password required in production mode")
    
    if salt is None:
        salt = generate_salt()
    
    return derive_key(password, salt), salt


# ==================== Encryption ====================

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """
    Encrypt data using AES-GCM.
    
    Args:
        data: Plaintext bytes
        key: 32-byte encryption key
        
    Returns:
        nonce + ciphertext (96-bit nonce prepended)
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext


def decrypt_data(encrypted: bytes, key: bytes) -> bytes:
    """
    Decrypt AES-GCM encrypted data.
    
    Args:
        encrypted: nonce + ciphertext
        key: 32-byte encryption key
        
    Returns:
        Decrypted plaintext
        
    Raises:
        Exception if decryption fails (wrong key or tampered data)
    """
    aesgcm = AESGCM(key)
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)


# ==================== Control File Management ====================

def _get_control_file_path(folder_path: str) -> str:
    """Get path to control file for a folder."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    return os.path.join(Config.CONTROL_DIR, f"{folder_name}.cybergaze")


def create_control_file(folder_path: str, 
                        user_id: str,
                        key: bytes) -> str:
    """
    Create encrypted control file for a protected folder.
    
    Args:
        folder_path: Path to folder being protected
        user_id: Owner's user ID
        key: Encryption key
        
    Returns:
        Path to created control file
    """
    control_data = {
        "folder_path": os.path.abspath(folder_path),
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "locked": True
    }
    
    plaintext = json.dumps(control_data).encode()
    encrypted = encrypt_data(plaintext, key)
    
    control_path = _get_control_file_path(folder_path)
    with open(control_path, "wb") as f:
        f.write(encrypted)
    
    return control_path


def verify_control_file(folder_path: str, key: bytes) -> Tuple[bool, Optional[dict]]:
    """
    Verify access by decrypting control file.
    
    Args:
        folder_path: Path to protected folder
        key: Encryption key
        
    Returns:
        Tuple of (success, control_data or None)
    """
    control_path = _get_control_file_path(folder_path)
    
    if not os.path.exists(control_path):
        return False, None
    
    try:
        with open(control_path, "rb") as f:
            encrypted = f.read()
        
        plaintext = decrypt_data(encrypted, key)
        control_data = json.loads(plaintext.decode())
        return True, control_data
    except Exception:
        return False, None


def update_control_file(folder_path: str, key: bytes, locked: bool) -> bool:
    """
    Update control file lock status.
    
    Args:
        folder_path: Path to protected folder
        key: Encryption key
        locked: New lock status
        
    Returns:
        Success status
    """
    success, control_data = verify_control_file(folder_path, key)
    if not success:
        return False
    
    control_data["locked"] = locked
    control_data["updated_at"] = datetime.now().isoformat()
    
    plaintext = json.dumps(control_data).encode()
    encrypted = encrypt_data(plaintext, key)
    
    control_path = _get_control_file_path(folder_path)
    with open(control_path, "wb") as f:
        f.write(encrypted)
    
    return True


# ==================== Folder Hide/Show ====================

def hide_folder(folder_path: str) -> bool:
    """
    Hide a folder (demo-level protection).
    
    Args:
        folder_path: Path to folder
        
    Returns:
        Success status
    """
    folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(folder_path):
        return False
    
    system = platform.system()
    
    try:
        if system == "Windows":
            # Use attrib command to set hidden attribute
            subprocess.run(
                ["attrib", "+h", "+s", folder_path],
                check=True,
                capture_output=True
            )
        else:
            # Unix: rename with dot prefix
            dirname = os.path.dirname(folder_path)
            basename = os.path.basename(folder_path)
            if not basename.startswith("."):
                new_path = os.path.join(dirname, f".{basename}")
                os.rename(folder_path, new_path)
        return True
    except Exception:
        return False


def show_folder(folder_path: str) -> bool:
    """
    Unhide a folder.
    
    Args:
        folder_path: Path to folder
        
    Returns:
        Success status
    """
    folder_path = os.path.abspath(folder_path)
    system = platform.system()
    
    try:
        if system == "Windows":
            # Try the path as-is first
            if os.path.exists(folder_path):
                subprocess.run(
                    ["attrib", "-h", "-s", folder_path],
                    check=True,
                    capture_output=True
                )
        else:
            # Unix: remove dot prefix
            dirname = os.path.dirname(folder_path)
            basename = os.path.basename(folder_path)
            if basename.startswith("."):
                new_path = os.path.join(dirname, basename[1:])
                os.rename(folder_path, new_path)
        return True
    except Exception:
        return False


# ==================== Rate Limiting ====================

class RateLimiter:
    """In-memory rate limiter for authentication attempts."""
    
    def __init__(self):
        self._attempts: Dict[str, list] = {}
        self._lockouts: Dict[str, datetime] = {}
    
    def is_locked_out(self, user_id: str) -> Tuple[bool, int]:
        """
        Check if user is locked out.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        if user_id not in self._lockouts:
            return False, 0
        
        lockout_end = self._lockouts[user_id]
        if datetime.now() >= lockout_end:
            del self._lockouts[user_id]
            self._attempts.pop(user_id, None)
            return False, 0
        
        remaining = (lockout_end - datetime.now()).seconds
        return True, remaining
    
    def record_attempt(self, user_id: str, success: bool) -> None:
        """
        Record an authentication attempt.
        
        Args:
            user_id: User identifier
            success: Whether attempt succeeded
        """
        if success:
            # Clear attempts on success
            self._attempts.pop(user_id, None)
            self._lockouts.pop(user_id, None)
            return
        
        # Record failed attempt
        if user_id not in self._attempts:
            self._attempts[user_id] = []
        
        self._attempts[user_id].append(datetime.now())
        
        # Clean old attempts (older than lockout duration)
        cutoff = datetime.now() - timedelta(seconds=Config.LOCKOUT_DURATION)
        self._attempts[user_id] = [
            t for t in self._attempts[user_id] if t > cutoff
        ]
        
        # Check if lockout needed
        if len(self._attempts[user_id]) >= Config.MAX_ATTEMPTS:
            self._lockouts[user_id] = datetime.now() + timedelta(
                seconds=Config.LOCKOUT_DURATION
            )
    
    def get_attempts_count(self, user_id: str) -> int:
        """Get current failed attempts count."""
        return len(self._attempts.get(user_id, []))


# Global rate limiter instance
rate_limiter = RateLimiter()


# ==================== Audit Logging ====================

def log_event(event_type: str, 
              user_id: str, 
              details: dict = None,
              success: bool = True) -> None:
    """
    Log an authentication event.
    
    Args:
        event_type: Type of event (e.g., "auth_attempt", "folder_lock")
        user_id: User identifier
        details: Additional event details
        success: Whether event was successful
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "success": success,
        "details": details or {}
    }
    
    log_file = os.path.join(Config.LOGS_DIR, f"{user_id}_audit.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_logs(user_id: str, limit: int = 50) -> list:
    """
    Retrieve audit logs for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum entries to return
        
    Returns:
        List of log entries (most recent first)
    """
    log_file = os.path.join(Config.LOGS_DIR, f"{user_id}_audit.jsonl")
    
    if not os.path.exists(log_file):
        return []
    
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    return logs[-limit:][::-1]  # Return most recent first


# ==================== Folder Registration ====================

def get_registered_folders(user_id: str) -> list:
    """
    Get list of folders registered by a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of registered folder paths
    """
    folders = []
    if os.path.exists(Config.CONTROL_DIR):
        for filename in os.listdir(Config.CONTROL_DIR):
            if filename.endswith(".cybergaze"):
                control_path = os.path.join(Config.CONTROL_DIR, filename)
                try:
                    # Try to read unencrypted header or check metadata
                    # For now, just list all control files
                    folders.append(filename[:-10])  # Remove .cybergaze
                except Exception:
                    pass
    return folders
