"""
CyberGaze - Flask Backend Application
REST API for face-based folder locking system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Import local modules
from database import (
    save_user_embedding, get_user_embedding, get_all_users,
    user_exists, delete_user, add_folder, update_folder_lock_status,
    get_folder, get_folder_by_path, get_all_folders, delete_folder,
    update_user_status, get_user_status, update_folder_encryption_status
)
from face_module import get_face_recognition
from folder_controller import (
    lock_folder, unlock_folder, generate_folder_id,
    get_folder_info, format_size, is_folder_locked
)
from encryption_engine import get_encryption_engine
from liveness_module import get_liveness_detector
from anti_spoof import get_spoof_detector
from audit_logger import get_audit_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Electron app communication

# Get face recognition instance
face_recognition = get_face_recognition()

# Get encryption engine instance
encryption_engine = get_encryption_engine()

# Get liveness detector and spoof detector instances
liveness_detector = get_liveness_detector()
spoof_detector = get_spoof_detector()

# Get audit logger instance
audit_logger = get_audit_logger()


# ============ Health Check ============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'CyberGaze Backend',
        'version': '1.0.0'
    })


# ============ Face Enrollment ============

@app.route('/enroll', methods=['POST'])
def enroll_face():
    """
    Enroll a new face or update existing enrollment.
    
    Request body:
    {
        "user_id": "string",
        "image": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        image_data = data.get('image')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        if not image_data:
            return jsonify({'success': False, 'error': 'image is required'}), 400
        
        # Extract face embedding
        embedding = face_recognition.extract_embedding_from_base64(image_data)
        
        if embedding is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image. Please ensure your face is clearly visible.'
            }), 400
        
        # Save embedding to database
        if save_user_embedding(user_id, embedding):
            is_new = not user_exists(user_id)
            audit_logger.log_event(audit_logger.EVENT_ENROLL, user_id=user_id, success=True)
            return jsonify({
                'success': True,
                'message': 'Face enrolled successfully' if is_new else 'Face updated successfully',
                'user_id': user_id
            })
        else:
            audit_logger.log_event(audit_logger.EVENT_ENROLL, user_id=user_id, success=False,
                                   details={'reason': 'Database save failed'})
            return jsonify({
                'success': False,
                'error': 'Failed to save face data'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Enrollment failed: {str(e)}'
        }), 500


# ============ Face Verification ============

@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Verify a face against enrolled user.
    
    Request body:
    {
        "user_id": "string",
        "image": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        image_data = data.get('image')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        if not image_data:
            return jsonify({'success': False, 'error': 'image is required'}), 400
        
        # Get stored embedding
        stored_embedding = get_user_embedding(user_id)
        
        if stored_embedding is None:
            return jsonify({
                'success': False,
                'error': 'User not enrolled. Please enroll first.'
            }), 404
        
        # Check lockout status
        lockout = audit_logger.is_locked_out(user_id)
        if lockout['locked']:
            return jsonify({
                'success': False,
                'error': lockout['message']
            }), 429
        
        # Extract embedding from live image
        live_embedding = face_recognition.extract_embedding_from_base64(image_data)
        
        if live_embedding is None:
            return jsonify({
                'success': False,
                'error': 'No face detected. Please position your face in the camera.'
            }), 400
        
        # Anti-spoof check on the live image
        try:
            import base64
            import io
            from PIL import Image
            import numpy as np
            
            b64_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(b64_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            frame_array = np.array(pil_image)
            import cv2
            bgr_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            
            spoof_result = spoof_detector.analyze_texture(bgr_frame)
            if not spoof_result['is_real']:
                audit_logger.log_event(audit_logger.EVENT_SPOOF, user_id=user_id,
                                       success=False, details={'spoof_score': spoof_result['spoof_score']})
                return jsonify({
                    'success': True,
                    'verified': False,
                    'spoof_detected': True,
                    'spoof_score': spoof_result['spoof_score'],
                    'message': 'Spoof detected! Please use a real face, not a photo or screen.'
                })
        except Exception as spoof_err:
            print(f"Anti-spoof check error (continuing): {spoof_err}")
        
        # Compare embeddings
        result = face_recognition.verify(stored_embedding, live_embedding)
        
        # Log verification attempt
        audit_logger.log_event(audit_logger.EVENT_VERIFY, user_id=user_id,
                               success=result['match'],
                               details={'similarity': result['similarity']})
        
        return jsonify({
            'success': True,
            'verified': result['match'],
            'similarity': result['similarity'],
            'threshold': result['threshold'],
            'message': 'Face verified successfully' if result['match'] else 'Face verification failed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Verification failed: {str(e)}'
        }), 500


# ============ Folder Management ============

@app.route('/lock-folder', methods=['POST'])
def lock_folder_endpoint():
    """
    Lock a folder.
    
    Request body:
    {
        "path": "folder_path",
        "owner_id": "user_id"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        folder_path = data.get('path')
        owner_id = data.get('owner_id')
        
        if not folder_path:
            return jsonify({'success': False, 'error': 'path is required'}), 400
        
        if not owner_id:
            return jsonify({'success': False, 'error': 'owner_id is required'}), 400
        
        # Check if user exists
        if not user_exists(owner_id):
            return jsonify({
                'success': False,
                'error': 'User not enrolled. Please enroll your face first.'
            }), 403
        
        # Normalize path
        folder_path = os.path.normpath(folder_path)
        
        # Derive encryption key from stored face embedding
        stored_embedding = get_user_embedding(owner_id)
        master_key = None
        if stored_embedding is not None:
            master_key = encryption_engine.derive_key(stored_embedding)
        
        # Check if folder is already in database
        existing_folder = get_folder_by_path(folder_path)
        
        if existing_folder:
            if existing_folder['is_locked']:
                return jsonify({
                    'success': False,
                    'error': 'Folder is already locked'
                }), 400
            
            # Lock existing folder (with encryption)
            result = lock_folder(folder_path, master_key=master_key)
            
            if result['success']:
                update_folder_lock_status(existing_folder['folder_id'], True)
                if master_key:
                    update_folder_encryption_status(existing_folder['folder_id'], True)
                
            return jsonify(result)
        
        # Add new folder and lock it
        folder_id = generate_folder_id()
        
        # Lock the folder (with encryption)
        result = lock_folder(folder_path, master_key=master_key)
        
        if result['success']:
            # Add to database
            add_folder(folder_id, folder_path, owner_id)
            update_folder_lock_status(folder_id, True)
            if master_key:
                update_folder_encryption_status(folder_id, True)
            result['folder_id'] = folder_id
            audit_logger.log_event(audit_logger.EVENT_LOCK, user_id=owner_id,
                                   folder_id=folder_id, success=True,
                                   details={'encrypted': master_key is not None})
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to lock folder: {str(e)}'
        }), 500


@app.route('/unlock-folder', methods=['POST'])
def unlock_folder_endpoint():
    """
    Unlock a folder (requires prior face verification).
    
    Request body:
    {
        "path": "folder_path",
        "user_id": "user_id",
        "verified": true  // Must be verified before calling
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        folder_path = data.get('path')
        user_id = data.get('user_id')
        verified = data.get('verified', False)
        
        if not folder_path:
            return jsonify({'success': False, 'error': 'path is required'}), 400
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        if not verified:
            return jsonify({
                'success': False,
                'error': 'Face verification required before unlocking'
            }), 403
        
        # Normalize path
        folder_path = os.path.normpath(folder_path)
        
        # Check folder in database
        folder_record = get_folder_by_path(folder_path)
        
        if folder_record:
            # Check ownership
            if folder_record['owner_id'] != user_id:
                return jsonify({
                    'success': False,
                    'error': 'You do not have permission to unlock this folder'
                }), 403
        
        # Derive decryption key from stored embedding
        stored_embedding = get_user_embedding(user_id)
        master_key = None
        if stored_embedding is not None:
            master_key = encryption_engine.derive_key(stored_embedding)
        
        # Unlock the folder (with decryption)
        result = unlock_folder(folder_path, master_key=master_key)
        
        if result['success'] and folder_record:
            update_folder_lock_status(folder_record['folder_id'], False)
            if master_key:
                update_folder_encryption_status(folder_record['folder_id'], False)
            audit_logger.log_event(audit_logger.EVENT_UNLOCK, user_id=user_id,
                                   folder_id=folder_record['folder_id'], success=True,
                                   details={'decrypted': master_key is not None})
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to unlock folder: {str(e)}'
        }), 500


@app.route('/get-folders', methods=['GET'])
def get_folders_endpoint():
    """
    Get all registered folders with their status.
    
    Query params:
    - owner_id (optional): Filter by owner
    """
    try:
        owner_id = request.args.get('owner_id')
        
        folders = get_all_folders(owner_id)
        
        # Enrich with current status
        enriched_folders = []
        for folder in folders:
            info = get_folder_info(folder['original_path'])
            enriched_folders.append({
                'folder_id': folder['folder_id'],
                'path': folder['original_path'],
                'name': os.path.basename(folder['original_path']),
                'is_locked': info['is_locked'],
                'exists': info['exists'],
                'size': format_size(info['size']),
                'owner_id': folder['owner_id'],
                'created_at': folder['created_at']
            })
        
        return jsonify({
            'success': True,
            'folders': enriched_folders,
            'count': len(enriched_folders)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get folders: {str(e)}'
        }), 500


@app.route('/add-folder', methods=['POST'])
def add_folder_endpoint():
    """
    Add a folder to track (without locking).
    
    Request body:
    {
        "path": "folder_path",
        "owner_id": "user_id"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        folder_path = data.get('path')
        owner_id = data.get('owner_id')
        
        if not folder_path:
            return jsonify({'success': False, 'error': 'path is required'}), 400
        
        if not owner_id:
            return jsonify({'success': False, 'error': 'owner_id is required'}), 400
        
        # Normalize path
        folder_path = os.path.normpath(folder_path)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'error': 'Folder does not exist'
            }), 404
        
        # Check if already tracked
        if get_folder_by_path(folder_path):
            return jsonify({
                'success': False,
                'error': 'Folder is already being tracked'
            }), 400
        
        folder_id = generate_folder_id()
        
        if add_folder(folder_id, folder_path, owner_id):
            return jsonify({
                'success': True,
                'folder_id': folder_id,
                'path': folder_path,
                'message': 'Folder added successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add folder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to add folder: {str(e)}'
        }), 500


@app.route('/remove-folder', methods=['POST'])
def remove_folder_endpoint():
    """
    Remove a folder from tracking (must be unlocked first).
    
    Request body:
    {
        "folder_id": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        folder_id = data.get('folder_id')
        
        if not folder_id:
            return jsonify({'success': False, 'error': 'folder_id is required'}), 400
        
        folder = get_folder(folder_id)
        
        if not folder:
            return jsonify({
                'success': False,
                'error': 'Folder not found'
            }), 404
        
        # Check if locked
        if folder['is_locked'] or is_folder_locked(folder['original_path']):
            return jsonify({
                'success': False,
                'error': 'Cannot remove a locked folder. Unlock it first.'
            }), 400
        
        if delete_folder(folder_id):
            return jsonify({
                'success': True,
                'message': 'Folder removed from tracking'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to remove folder'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to remove folder: {str(e)}'
        }), 500


# ============ User Management ============

@app.route('/users', methods=['GET'])
def get_users():
    """Get all enrolled users"""
    try:
        users = get_all_users()
        return jsonify({
            'success': True,
            'users': users,
            'count': len(users)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user_endpoint(user_id):
    """Delete a user and their data"""
    try:
        if delete_user(user_id):
            return jsonify({
                'success': True,
                'message': 'User deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/users/<user_id>/exists', methods=['GET'])
def check_user_exists(user_id):
    """Check if a user is enrolled"""
    exists = user_exists(user_id)
    return jsonify({
        'success': True,
        'exists': exists,
        'user_id': user_id
    })


# ============ Liveness Detection ============

@app.route('/get-challenge', methods=['GET'])
def get_challenge():
    """
    Generate a random liveness challenge.
    
    Returns:
    {
        "challenge_id": "uuid",
        "challenge": "blink_twice|turn_left|turn_right",
        "instruction": "human-readable instruction",
        "duration_seconds": 3
    }
    """
    try:
        challenge = liveness_detector.generate_challenge()
        return jsonify({
            'success': True,
            **challenge
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate challenge: {str(e)}'
        }), 500


@app.route('/verify-liveness', methods=['POST'])
def verify_liveness():
    """
    Verify liveness challenge with captured frames.
    
    Request body:
    {
        "challenge_id": "uuid",
        "frames": ["base64_frame1", "base64_frame2", ...]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        challenge_id = data.get('challenge_id')
        frames = data.get('frames', [])
        
        if not challenge_id:
            return jsonify({'success': False, 'error': 'challenge_id is required'}), 400
        
        if not frames or len(frames) < 5:
            return jsonify({
                'success': False,
                'error': 'At least 5 frames required for liveness verification'
            }), 400
        
        # Verify the liveness challenge
        result = liveness_detector.verify_challenge(challenge_id, frames)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Liveness verification failed: {str(e)}'
        }), 500


@app.route('/security-status', methods=['GET'])
def security_status():
    """Get security module status."""
    threat = audit_logger.analyze_threats()
    return jsonify({
        'success': True,
        'liveness_available': liveness_detector.is_available,
        'encryption_available': True,
        'anti_spoof_available': True,
        'threat_analysis': threat
    })


@app.route('/audit-logs', methods=['GET'])
def get_audit_logs():
    """Get paginated audit log entries."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    event_type = request.args.get('event_type', None)
    user_id = request.args.get('user_id', None)
    
    logs = audit_logger.get_audit_logs(
        limit=limit, offset=offset,
        event_type=event_type, user_id=user_id
    )
    return jsonify({'success': True, **logs})


@app.route('/threat-analysis', methods=['GET'])
def get_threat_analysis():
    """Get threat analysis report."""
    user_id = request.args.get('user_id', None)
    analysis = audit_logger.analyze_threats(user_id)
    return jsonify({'success': True, **analysis})

# ============ Main Entry Point ============

if __name__ == '__main__':
    print("=" * 50)
    print("CyberGaze Backend Server")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print(f"Liveness Detection: {'Available' if liveness_detector.is_available else 'Limited (dlib not installed)'}")
    print(f"Anti-Spoofing: Available (LBP texture analysis)")
    print(f"Encryption: Available (AES-256-GCM)")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
