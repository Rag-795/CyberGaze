"""
CyberGaze - Folder Controller Module
Handles folder locking/unlocking using file system operations
"""

import os
import shutil
import platform
import uuid

# Import encryption engine (lazy to avoid circular imports)
def _get_encryption_engine():
    from encryption_engine import get_encryption_engine
    return get_encryption_engine()


# Prefix used to mark locked folders
LOCK_PREFIX = ".cybergaze_locked_"


def is_valid_folder(path: str) -> bool:
    """Check if path is a valid, existing folder"""
    return os.path.exists(path) and os.path.isdir(path)


def get_locked_path(original_path: str) -> str:
    """
    Generate the locked path for a folder.
    Adds a hidden prefix to the folder name.
    """
    parent_dir = os.path.dirname(original_path)
    folder_name = os.path.basename(original_path)
    locked_name = f"{LOCK_PREFIX}{folder_name}"
    return os.path.join(parent_dir, locked_name)


def get_original_path(locked_path: str) -> str:
    """
    Get original path from a locked path.
    Removes the lock prefix from the folder name.
    """
    parent_dir = os.path.dirname(locked_path)
    folder_name = os.path.basename(locked_path)
    
    if folder_name.startswith(LOCK_PREFIX):
        original_name = folder_name[len(LOCK_PREFIX):]
        return os.path.join(parent_dir, original_name)
    
    return locked_path


def is_folder_locked(path: str) -> bool:
    """Check if a folder is currently locked (by checking if locked version exists)"""
    locked_path = get_locked_path(path)
    return os.path.exists(locked_path)


def lock_folder(original_path: str, master_key: bytes = None) -> dict:
    """
    Lock a folder by renaming it with a hidden prefix.
    If master_key is provided, encrypts all files before renaming.
    
    Args:
        original_path: Path to the folder to lock
        master_key: Optional 32-byte AES key for encryption
        
    Returns:
        dict with success status and details
    """
    try:
        # Validate path
        if not is_valid_folder(original_path):
            return {
                'success': False,
                'error': f"Folder does not exist: {original_path}"
            }
        
        # Check if already locked
        if is_folder_locked(original_path):
            return {
                'success': False,
                'error': "Folder is already locked"
            }
        
        locked_path = get_locked_path(original_path)
        
        # Check if locked path already exists (conflict)
        if os.path.exists(locked_path):
            return {
                'success': False,
                'error': "A locked folder with this name already exists"
            }
        
        # Encrypt files if master key provided
        encryption_result = None
        if master_key:
            try:
                engine = _get_encryption_engine()
                encryption_result = engine.encrypt_folder(original_path, master_key)
                if not encryption_result['success']:
                    return {
                        'success': False,
                        'error': f"Encryption failed: {encryption_result['errors']}"
                    }
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Encryption failed: {str(e)}"
                }
        
        # Rename folder to lock it
        os.rename(original_path, locked_path)
        
        # On Windows, set hidden attribute
        if platform.system() == 'Windows':
            try:
                import ctypes
                FILE_ATTRIBUTE_HIDDEN = 0x02
                ctypes.windll.kernel32.SetFileAttributesW(locked_path, FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                print(f"Could not set hidden attribute: {e}")
        
        result = {
            'success': True,
            'original_path': original_path,
            'locked_path': locked_path,
            'message': "Folder locked successfully",
            'encrypted': master_key is not None
        }
        
        if encryption_result:
            result['encrypted_files'] = encryption_result['encrypted_count']
        
        return result
        
    except PermissionError:
        return {
            'success': False,
            'error': "Permission denied. Make sure no files in the folder are in use."
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to lock folder: {str(e)}"
        }


def unlock_folder(original_path: str, master_key: bytes = None) -> dict:
    """
    Unlock a folder by restoring its original name.
    If master_key is provided, decrypts all files after renaming.
    
    Args:
        original_path: Original path of the folder (before locking)
        master_key: Optional 32-byte AES key for decryption
        
    Returns:
        dict with success status and details
    """
    try:
        locked_path = get_locked_path(original_path)
        
        # Check if folder is actually locked
        if not os.path.exists(locked_path):
            # Maybe already unlocked?
            if os.path.exists(original_path):
                return {
                    'success': True,
                    'message': "Folder is already unlocked"
                }
            return {
                'success': False,
                'error': f"Locked folder not found: {locked_path}"
            }
        
        # Check if original path is available
        if os.path.exists(original_path):
            return {
                'success': False,
                'error': "Cannot unlock: a folder with the original name already exists"
            }
        
        # Remove hidden attribute on Windows before renaming
        if platform.system() == 'Windows':
            try:
                import ctypes
                FILE_ATTRIBUTE_NORMAL = 0x80
                ctypes.windll.kernel32.SetFileAttributesW(locked_path, FILE_ATTRIBUTE_NORMAL)
            except Exception as e:
                print(f"Could not remove hidden attribute: {e}")
        
        # Rename folder to unlock it
        os.rename(locked_path, original_path)
        
        # Decrypt files if master key provided
        decryption_result = None
        if master_key:
            try:
                engine = _get_encryption_engine()
                decryption_result = engine.decrypt_folder(original_path, master_key)
                if not decryption_result['success']:
                    print(f"Warning: Some files failed to decrypt: {decryption_result['errors']}")
            except Exception as e:
                print(f"Warning: Decryption failed: {str(e)}")
        
        result = {
            'success': True,
            'original_path': original_path,
            'message': "Folder unlocked successfully",
            'decrypted': master_key is not None
        }
        
        if decryption_result:
            result['decrypted_files'] = decryption_result['decrypted_count']
        
        return result
        
    except PermissionError:
        return {
            'success': False,
            'error': "Permission denied. Make sure no files in the folder are in use."
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to unlock folder: {str(e)}"
        }


def generate_folder_id() -> str:
    """Generate a unique folder ID"""
    return str(uuid.uuid4())


def get_folder_info(path: str) -> dict:
    """
    Get information about a folder.
    
    Args:
        path: Path to the folder
        
    Returns:
        dict with folder information
    """
    locked_path = get_locked_path(path)
    
    if os.path.exists(path):
        return {
            'path': path,
            'name': os.path.basename(path),
            'exists': True,
            'is_locked': False,
            'size': get_folder_size(path)
        }
    elif os.path.exists(locked_path):
        return {
            'path': path,
            'name': os.path.basename(path),
            'exists': True,
            'is_locked': True,
            'size': get_folder_size(locked_path)
        }
    else:
        return {
            'path': path,
            'name': os.path.basename(path),
            'exists': False,
            'is_locked': False,
            'size': 0
        }


def get_folder_size(path: str) -> int:
    """Calculate total size of a folder in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    pass
    except Exception:
        pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
