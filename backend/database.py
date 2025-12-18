"""
CyberGaze - Database Module
SQLite database for storing face embeddings and folder metadata
"""

import sqlite3
import os
import json
import numpy as np
from datetime import datetime

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'cybergaze.db')


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table - stores face embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Folders table - stores folder metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id TEXT UNIQUE NOT NULL,
            original_path TEXT NOT NULL,
            is_locked INTEGER DEFAULT 0,
            owner_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DATABASE_PATH}")


# ============ User Operations ============

def save_user_embedding(user_id: str, embedding: np.ndarray) -> bool:
    """Save or update user face embedding"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Convert numpy array to JSON string for storage
        embedding_json = json.dumps(embedding.tolist())
        
        # Try to update existing user first
        cursor.execute('''
            UPDATE users SET embedding = ?, updated_at = ? WHERE user_id = ?
        ''', (embedding_json, datetime.now(), user_id))
        
        if cursor.rowcount == 0:
            # Insert new user if not exists
            cursor.execute('''
                INSERT INTO users (user_id, embedding) VALUES (?, ?)
            ''', (user_id, embedding_json))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving user embedding: {e}")
        return False
    finally:
        conn.close()


def get_user_embedding(user_id: str) -> np.ndarray:
    """Get user face embedding by user_id"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT embedding FROM users WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return np.array(json.loads(row['embedding']))
    return None


def get_all_users():
    """Get all enrolled users"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT user_id, created_at, updated_at FROM users')
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def delete_user(user_id: str) -> bool:
    """Delete user and their folders"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM folders WHERE owner_id = ?', (user_id,))
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False
    finally:
        conn.close()


def user_exists(user_id: str) -> bool:
    """Check if user exists"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    
    return exists


# ============ Folder Operations ============

def add_folder(folder_id: str, original_path: str, owner_id: str) -> bool:
    """Add a new folder to track"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO folders (folder_id, original_path, owner_id)
            VALUES (?, ?, ?)
        ''', (folder_id, original_path, owner_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Folder already exists
        return False
    except Exception as e:
        print(f"Error adding folder: {e}")
        return False
    finally:
        conn.close()


def update_folder_lock_status(folder_id: str, is_locked: bool) -> bool:
    """Update folder lock status"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE folders SET is_locked = ? WHERE folder_id = ?
        ''', (1 if is_locked else 0, folder_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating folder status: {e}")
        return False
    finally:
        conn.close()


def get_folder(folder_id: str) -> dict:
    """Get folder by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM folders WHERE folder_id = ?', (folder_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def get_folder_by_path(original_path: str) -> dict:
    """Get folder by original path"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM folders WHERE original_path = ?', (original_path,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def get_all_folders(owner_id: str = None) -> list:
    """Get all folders, optionally filtered by owner"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if owner_id:
        cursor.execute('SELECT * FROM folders WHERE owner_id = ?', (owner_id,))
    else:
        cursor.execute('SELECT * FROM folders')
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def delete_folder(folder_id: str) -> bool:
    """Delete folder from tracking"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM folders WHERE folder_id = ?', (folder_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting folder: {e}")
        return False
    finally:
        conn.close()


# Initialize database on module import
init_database()
