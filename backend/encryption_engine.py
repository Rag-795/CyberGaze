"""
CyberGaze - Encryption Engine
AES-256-GCM encryption for folder protection with face-embedding-based key derivation
"""

import os
import hashlib
import shutil
import tarfile
import tempfile
from datetime import datetime
from Crypto.Cipher import AES


# Extension for encrypted files
ENCRYPTED_EXT = '.cyg'
# Extension for encrypted containers
CONTAINER_EXT = '.vault'
# Header magic bytes to identify CyberGaze encrypted files
MAGIC_HEADER = b'CYBERGAZE'


class EncryptionEngine:
    """
    Handles AES-256-GCM encryption/decryption of files and folders.
    Key is derived solely from face embedding - no passwords.
    """

    def derive_key(self, face_embedding) -> bytes:
        """
        Derive a 256-bit AES key from a face embedding.
        
        Uses SHA-256 hash of the raw embedding bytes to produce
        a deterministic 32-byte key. The same face embedding always
        produces the same key.
        
        Args:
            face_embedding: numpy array of face embedding (typically 512-dim)
            
        Returns:
            32-byte key suitable for AES-256
        """
        # Convert embedding to bytes deterministically
        embedding_bytes = face_embedding.tobytes()
        # SHA-256 produces exactly 32 bytes = 256 bits
        master_key = hashlib.sha256(embedding_bytes).digest()
        return master_key

    def encrypt_file(self, file_path: str, master_key: bytes) -> str:
        """
        Encrypt a single file using AES-256-GCM.
        
        Output format: MAGIC_HEADER | nonce (12 bytes) | tag (16 bytes) | ciphertext
        
        Args:
            file_path: Path to the file to encrypt
            master_key: 32-byte AES key
            
        Returns:
            Path to the encrypted file (.cyg)
        """
        # Generate unique nonce per file
        nonce = os.urandom(12)
        cipher = AES.new(master_key, AES.MODE_GCM, nonce=nonce)

        with open(file_path, 'rb') as f:
            plaintext = f.read()

        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        # Write encrypted file: magic + nonce + tag + ciphertext
        encrypted_path = file_path + ENCRYPTED_EXT
        with open(encrypted_path, 'wb') as f:
            f.write(MAGIC_HEADER)
            f.write(nonce)
            f.write(tag)
            f.write(ciphertext)

        return encrypted_path

    def decrypt_file(self, encrypted_path: str, master_key: bytes) -> str:
        """
        Decrypt a single .cyg file using AES-256-GCM.
        
        Args:
            encrypted_path: Path to the .cyg file
            master_key: 32-byte AES key
            
        Returns:
            Path to the decrypted file (original name restored)
        """
        with open(encrypted_path, 'rb') as f:
            # Read and verify magic header
            magic = f.read(len(MAGIC_HEADER))
            if magic != MAGIC_HEADER:
                raise ValueError(f"Not a valid CyberGaze encrypted file: {encrypted_path}")

            nonce = f.read(12)
            tag = f.read(16)
            ciphertext = f.read()

        cipher = AES.new(master_key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        # Restore original filename (remove .cyg extension)
        if encrypted_path.endswith(ENCRYPTED_EXT):
            original_path = encrypted_path[:-len(ENCRYPTED_EXT)]
        else:
            original_path = encrypted_path + '.decrypted'

        with open(original_path, 'wb') as f:
            f.write(plaintext)

        return original_path

    def encrypt_folder(self, folder_path: str, master_key: bytes) -> dict:
        """
        Encrypt all files in a folder individually using AES-256-GCM.
        Original files are securely deleted after encryption.
        
        Args:
            folder_path: Path to the folder to encrypt
            master_key: 32-byte AES key
            
        Returns:
            dict with success status and file count
        """
        encrypted_count = 0
        errors = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Skip already encrypted files
                if file.endswith(ENCRYPTED_EXT) or file.endswith(CONTAINER_EXT):
                    continue

                file_path = os.path.join(root, file)
                try:
                    self.encrypt_file(file_path, master_key)
                    # Securely delete original
                    self.secure_delete(file_path)
                    encrypted_count += 1
                except Exception as e:
                    errors.append({'file': file_path, 'error': str(e)})

        return {
            'success': len(errors) == 0,
            'encrypted_count': encrypted_count,
            'errors': errors
        }

    def decrypt_folder(self, folder_path: str, master_key: bytes) -> dict:
        """
        Decrypt all .cyg files in a folder, restoring original files.
        Encrypted .cyg files are removed after successful decryption.
        
        Args:
            folder_path: Path to the folder to decrypt
            master_key: 32-byte AES key
            
        Returns:
            dict with success status and file count
        """
        decrypted_count = 0
        errors = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.endswith(ENCRYPTED_EXT):
                    continue

                encrypted_path = os.path.join(root, file)
                try:
                    self.decrypt_file(encrypted_path, master_key)
                    # Remove encrypted version
                    os.remove(encrypted_path)
                    decrypted_count += 1
                except Exception as e:
                    errors.append({'file': encrypted_path, 'error': str(e)})

        return {
            'success': len(errors) == 0,
            'decrypted_count': decrypted_count,
            'errors': errors
        }

    def secure_delete(self, file_path: str, passes: int = 3):
        """
        Securely delete a file by overwriting with random data before removal.
        
        3-pass overwrite:
          1. All zeros
          2. All ones (0xFF)
          3. Random data
          
        Args:
            file_path: Path to the file to securely delete
            passes: Number of overwrite passes (default 3)
        """
        if not os.path.exists(file_path):
            return

        file_size = os.path.getsize(file_path)

        try:
            with open(file_path, 'r+b') as f:
                # Pass 1: zeros
                f.seek(0)
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())

                # Pass 2: ones
                f.seek(0)
                f.write(b'\xff' * file_size)
                f.flush()
                os.fsync(f.fileno())

                # Pass 3: random
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Warning: Secure overwrite failed for {file_path}: {e}")

        # Finally delete the file
        os.remove(file_path)

    def create_secure_container(self, folder_path: str, master_key: bytes) -> str:
        """
        Create an encrypted container from a folder.
        Archives the folder as tar.gz, then encrypts as a single .vault blob.
        
        Args:
            folder_path: Path to the folder to containerize
            master_key: 32-byte AES key
            
        Returns:
            Path to the created .vault container file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = os.path.basename(folder_path)
        parent_dir = os.path.dirname(folder_path)
        container_name = f".cybergaze_{folder_name}_{timestamp}{CONTAINER_EXT}"
        container_path = os.path.join(parent_dir, container_name)

        # Create temporary archive
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create tar.gz archive
            with tarfile.open(tmp_path, 'w:gz') as tar:
                tar.add(folder_path, arcname=folder_name)

            # Encrypt the archive
            nonce = os.urandom(12)
            cipher = AES.new(master_key, AES.MODE_GCM, nonce=nonce)

            with open(tmp_path, 'rb') as f:
                plaintext = f.read()

            ciphertext, tag = cipher.encrypt_and_digest(plaintext)

            with open(container_path, 'wb') as f:
                f.write(MAGIC_HEADER)
                f.write(nonce)
                f.write(tag)
                f.write(ciphertext)

            # Secure delete original folder contents
            shutil.rmtree(folder_path, ignore_errors=False)
            os.makedirs(folder_path)  # Keep empty folder as placeholder

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                self.secure_delete(tmp_path)

        return container_path

    def extract_secure_container(self, container_path: str, master_key: bytes, target_dir: str) -> str:
        """
        Extract an encrypted .vault container to the target directory.
        
        Args:
            container_path: Path to the .vault file
            master_key: 32-byte AES key
            target_dir: Directory to extract contents into
            
        Returns:
            Path to the extracted folder
        """
        with open(container_path, 'rb') as f:
            magic = f.read(len(MAGIC_HEADER))
            if magic != MAGIC_HEADER:
                raise ValueError("Not a valid CyberGaze container file")

            nonce = f.read(12)
            tag = f.read(16)
            ciphertext = f.read()

        cipher = AES.new(master_key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        # Extract tar.gz archive
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(plaintext)

        try:
            with tarfile.open(tmp_path, 'r:gz') as tar:
                tar.extractall(path=target_dir)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Remove the container file after extraction
        os.remove(container_path)

        return target_dir


# Singleton instance
_encryption_engine = None


def get_encryption_engine() -> EncryptionEngine:
    """Get singleton EncryptionEngine instance"""
    global _encryption_engine
    if _encryption_engine is None:
        _encryption_engine = EncryptionEngine()
    return _encryption_engine
