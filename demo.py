"""
CyberGaze Demo v2 - Improved Enrollment & Recognition
Better enrollment with variation, stricter matching.

Usage:
    python demo.py
    
Controls:
    E - Enroll face (guided multi-pose capture)
    A - Authenticate (with liveness challenge)
    C - Switch liveness challenge
    D - Toggle distance gating
    R - Re-enroll (delete and start fresh)
    Q - Quit
"""
import cv2
import numpy as np
import time
import os
from config import Config
from modules.preprocessing import preprocess, quality_check, to_grayscale
from modules.detection import detect_faces, detect_eyes, get_largest_face
from modules.alignment import extract_aligned_face
from modules.recognition import (
    train_recognizer, save_model, load_model, recognize, get_enrolled_users
)
from modules.liveness import Challenge, detect_blink, detect_head_turn, extract_eye_rois
from modules.calibration import get_calibration_matrix, estimate_distance

# Initialize
Config.init_directories()


class DemoState:
    """Track demo state."""
    def __init__(self):
        self.mode = "IDLE"
        self.enrolled = False
        self.recognizer = None
        self.challenge = Challenge.BLINK
        self.distance_gating = False  # Start with distance gating OFF for easier testing
        
        # Enrollment state - now with phases
        self.enroll_samples = []
        self.enroll_phase = 0  # 0=center, 1=left, 2=right, 3=up, 4=down
        self.enroll_phases = ["CENTER", "TURN LEFT", "TURN RIGHT", "LOOK UP", "LOOK DOWN"]
        self.samples_per_phase = 3
        self.last_capture_time = 0
        self.capture_delay = 0.5  # 500ms between captures
        
        # Auth state
        self.auth_frames = []
        self.auth_faces = []
        self.auth_eyes = []
        self.auth_start_time = 0
        self.auth_duration = 5
        
        # Results
        self.last_result = ""
        self.result_time = 0
        self.last_confidence = 0
        
        # Load existing model
        self._load_existing()
    
    def _load_existing(self):
        users = get_enrolled_users()
        if users:
            self.recognizer = load_model(users[0])
            self.enrolled = True
            self.last_result = f"Loaded: {users[0]}"
            self.result_time = time.time()
    
    def reset_enrollment(self):
        """Reset enrollment state."""
        self.enroll_samples = []
        self.enroll_phase = 0
        self.last_capture_time = 0
    
    @property
    def current_phase_name(self):
        if self.enroll_phase < len(self.enroll_phases):
            return self.enroll_phases[self.enroll_phase]
        return "DONE"
    
    @property
    def samples_in_phase(self):
        start = self.enroll_phase * self.samples_per_phase
        end = min(len(self.enroll_samples), (self.enroll_phase + 1) * self.samples_per_phase)
        return end - start
    
    @property
    def total_samples_needed(self):
        return len(self.enroll_phases) * self.samples_per_phase


def draw_overlay(frame, state, face, distance, quality_ok, fps):
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "CyberGaze - CV Demo", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Mode status
    if state.mode == "IDLE":
        status_text = "Ready"
        status_color = (0, 255, 0)
    elif state.mode == "ENROLLING":
        phase_samples = state.samples_in_phase
        status_text = f"Enrolling: {state.current_phase_name} ({phase_samples}/{state.samples_per_phase})"
        status_color = (255, 255, 0)
    else:
        remaining = max(0, state.auth_duration - (time.time() - state.auth_start_time))
        status_text = f"Authenticating... {remaining:.1f}s"
        status_color = (0, 255, 255)
    
    cv2.putText(frame, status_text, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Progress bar for enrollment
    if state.mode == "ENROLLING":
        bar_w = 300
        bar_h = 15
        bar_x = 15
        bar_y = 75
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        
        # Progress
        progress = len(state.enroll_samples) / state.total_samples_needed
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_w * progress), bar_y + bar_h), 
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(frame, f"{len(state.enroll_samples)}/{state.total_samples_needed}", 
                   (bar_x + bar_w + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Right side info
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    challenge_color = (0, 200, 255)
    cv2.putText(frame, f"Challenge: {state.challenge.value}", (w - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, challenge_color, 1)
    
    # Threshold info
    cv2.putText(frame, f"Threshold: {Config.LBPH_THRESHOLD}", (w - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    enrolled_text = "Enrolled: YES" if state.enrolled else "Enrolled: NO"
    cv2.putText(frame, enrolled_text, (w - 140, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0) if state.enrolled else (0, 100, 255), 1)
    
    # Bottom bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, "E=Enroll  A=Auth  C=Challenge  D=Distance  R=Re-enroll  Q=Quit", 
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    # Face distance info
    if face and distance:
        x, y, fw, fh = face
        dist_text = f"{distance:.2f}m"
        in_range = Config.Z_MIN <= distance <= Config.Z_MAX
        dist_color = (0, 255, 0) if in_range else (0, 165, 255)
        
        cv2.putText(frame, dist_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)
    
    # Quality indicator  
    if state.mode != "IDLE":
        quality_text = "Quality: OK" if quality_ok else "Quality: POOR"
        quality_color = (0, 255, 0) if quality_ok else (0, 0, 255)
        cv2.putText(frame, quality_text, (w - 130, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
    
    # Result message
    if state.last_result and (time.time() - state.result_time) < 4:
        # Determine color based on result
        if "AUTHENTICATED" in state.last_result or "Complete" in state.last_result:
            box_color = (0, 180, 0)
        elif "FAILED" in state.last_result or "NOT" in state.last_result:
            box_color = (0, 0, 200)
        else:
            box_color = (200, 150, 0)
        
        # Result box
        box_w = 400
        box_h = 60
        box_x = (w - box_w) // 2
        box_y = h // 2 - 30
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 3)
        
        # Main text
        text_size = cv2.getTextSize(state.last_result, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, state.last_result, (text_x, box_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence sub-text
        if state.last_confidence > 0:
            conf_text = f"Confidence: {state.last_confidence:.1f} (threshold: {Config.LBPH_THRESHOLD})"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            conf_x = (w - conf_size[0]) // 2
            cv2.putText(frame, conf_text, (conf_x, box_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    return frame


def draw_face_box(frame, face, eyes, color=(0, 255, 0)):
    """Draw face bounding box and eye markers."""
    if not face:
        return frame
    
    x, y, w, h = face
    
    # Full rectangle for clarity
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Corner accents
    corner_len = 20
    thickness = 3
    
    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
    
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
    
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
    
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
    
    # Eyes
    for (ex, ey, ew, eh) in eyes:
        center = (x + ex + ew // 2, y + ey + eh // 2)
        cv2.circle(frame, center, 5, (255, 0, 255), -1)
    
    return frame


def draw_enrollment_guide(frame, phase_name):
    """Draw guide arrows for enrollment poses."""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    arrow_color = (0, 255, 255)
    arrow_len = 80
    
    # Draw instruction
    cv2.putText(frame, f"Look: {phase_name}", (center_x - 80, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, arrow_color, 2)
    
    if phase_name == "CENTER":
        # Circle in center
        cv2.circle(frame, (center_x, center_y), 30, arrow_color, 3)
        cv2.putText(frame, "Face the camera", (center_x - 90, center_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
    
    elif phase_name == "TURN LEFT":
        # Arrow pointing left
        cv2.arrowedLine(frame, (center_x + 50, center_y), 
                       (center_x - arrow_len, center_y), arrow_color, 4, tipLength=0.3)
    
    elif phase_name == "TURN RIGHT":
        # Arrow pointing right
        cv2.arrowedLine(frame, (center_x - 50, center_y), 
                       (center_x + arrow_len, center_y), arrow_color, 4, tipLength=0.3)
    
    elif phase_name == "LOOK UP":
        # Arrow pointing up
        cv2.arrowedLine(frame, (center_x, center_y + 50), 
                       (center_x, center_y - arrow_len), arrow_color, 4, tipLength=0.3)
    
    elif phase_name == "LOOK DOWN":
        # Arrow pointing down
        cv2.arrowedLine(frame, (center_x, center_y - 50), 
                       (center_x, center_y + arrow_len), arrow_color, 4, tipLength=0.3)
    
    return frame


def delete_enrolled_user():
    """Delete existing enrolled user model."""
    users = get_enrolled_users()
    for user in users:
        model_path = os.path.join(Config.TEMPLATES_DIR, f"{user}.yml")
        meta_path = os.path.join(Config.TEMPLATES_DIR, f"{user}_meta.json")
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
    print("✓ Deleted existing enrollment")


def main():
    """Main demo loop."""
    print("\n" + "="*60)
    print("  CyberGaze - Computer Vision Demo")
    print("  Face Recognition with Multi-Pose Enrollment")
    print("="*60)
    print(f"\nRecognition threshold: {Config.LBPH_THRESHOLD}")
    print("  (Lower confidence = better match)")
    print("  (Match accepted if confidence < threshold)")
    print("\nControls:")
    print("  E - Enroll face (multi-pose: center, left, right, up, down)")
    print("  A - Authenticate with liveness challenge")
    print("  C - Switch liveness challenge")
    print("  D - Toggle distance gating")
    print("  R - Re-enroll (delete existing and start fresh)")
    print("  Q - Quit")
    print("\n" + "="*60 + "\n")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    state = DemoState()
    prev_time = time.time()
    fps = 0
    
    challenges = [Challenge.BLINK, Challenge.LOOK_LEFT, Challenge.LOOK_RIGHT]
    challenge_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        
        gray = to_grayscale(frame)
        gray_processed = preprocess(gray, mode="detect")
        gray_liveness  = preprocess(gray, mode="liveness")
        
        quality_ok, _ = quality_check(gray)
        K = get_calibration_matrix(gray.shape)
        
        faces = detect_faces(gray_processed)
        face = get_largest_face(faces)
        
        distance = None
        eyes = []
        
        if face:
            x, y, w, h = face
            distance = estimate_distance(K, w)
            eyes = detect_eyes(gray_processed, face)
            
            # Color based on mode and distance
            color = (0, 255, 0)
            if state.distance_gating and not (Config.Z_MIN <= distance <= Config.Z_MAX):
                color = (0, 165, 255)
            if state.mode == "ENROLLING":
                color = (255, 255, 0)
            elif state.mode == "AUTHENTICATING":
                color = (0, 255, 255)
            
            frame = draw_face_box(frame, face, eyes, color)
        
        # Handle enrollment with phases
        if state.mode == "ENROLLING":
            frame = draw_enrollment_guide(frame, state.current_phase_name)
            
            if face and (curr_time - state.last_capture_time) >= state.capture_delay:
                in_range = not state.distance_gating or (Config.Z_MIN <= distance <= Config.Z_MAX)
                
                if in_range:
                    aligned, info = extract_aligned_face(gray_processed, face)
                    state.enroll_samples.append(aligned)
                    state.last_capture_time = curr_time
                    
                    # Flash effect
                    frame = cv2.addWeighted(frame, 0.5, np.ones_like(frame) * 255, 0.5, 0)
                    
                    # Check if phase complete
                    if state.samples_in_phase >= state.samples_per_phase:
                        state.enroll_phase += 1
                        print(f"  Phase {state.enroll_phase}/{len(state.enroll_phases)} complete")
                    
                    # Check if enrollment complete
                    if len(state.enroll_samples) >= state.total_samples_needed:
                        state.recognizer = train_recognizer(state.enroll_samples, label=0)
                        save_model(state.recognizer, "demo_user")
                        state.enrolled = True
                        state.mode = "IDLE"
                        state.last_result = "Enrollment Complete!"
                        state.result_time = time.time()
                        state.enroll_samples = []
                        print(f"✓ Enrolled with {state.total_samples_needed} samples across {len(state.enroll_phases)} poses")
        
        # Handle authentication
        if state.mode == "AUTHENTICATING":
            if face:
                in_range = (not state.distance_gating) or (Config.Z_MIN <= distance <= Config.Z_MAX)
                # Mark frame quality for later filtering
                is_good = quality_ok and in_range
                
                # Debug: log first few failures
                if not is_good and len(state.auth_frames) < 3:
                    _, q_details = quality_check(gray)
                    # print(f"  Frame rejected: quality={quality_ok}, in_range={in_range}, distance={distance:.2f}m, details={q_details}")
                
                state.auth_frames.append((gray_liveness.copy(), is_good))
                state.auth_faces.append(face)
                state.auth_eyes.append(eyes)
            
            if (curr_time - state.auth_start_time) >= state.auth_duration:
                # Filter to only use good quality frames for liveness
                good_frames = [(f, fa, ey) for (f, is_good), fa, ey in zip(state.auth_frames, state.auth_faces, state.auth_eyes) if is_good]
                total_frames = len(state.auth_frames)
                good_ratio = len(good_frames) / max(total_frames, 1)
                
                # Adaptive: if distance gating is ON and we have few good frames, try using all frames
                if len(good_frames) < 5 and total_frames >= 15:
                    # print(f"  Warning: Only {len(good_frames)} good frames, using all {total_frames} frames")
                    # Use all frames but extract just the frame data
                    good_frames = [(f, fa, ey) for (f, is_good), fa, ey in zip(state.auth_frames, state.auth_faces, state.auth_eyes)]
                
                if len(good_frames) >= 5:
                    # Liveness check
                    if state.challenge == Challenge.BLINK:
                        eye_rois = []
                        for f, fa, ey in good_frames:
                            if fa and len(ey) >= 1:
                                ey_sorted = sorted(ey, key=lambda e: e[0])
                                rois = extract_eye_rois(f, fa, ey_sorted[:2])
                                if len(rois) == 2:
                                    # combine by averaging scores later; simplest: stack into one image
                                    # OR just append both and lower required_blinks accordingly
                                    eye_rois.append(rois[0])
                                    eye_rois.append(rois[1])
                                elif len(rois) == 1:
                                    eye_rois.append(rois[0])

                        # If you append both eyes as separate frames, set required_blinks=1 (because you doubled samples)
                        liveness_ok = len(eye_rois) >= 20 and detect_blink(eye_rois, required_blinks=1)[0]
                    else:
                        direction = "left" if state.challenge == Challenge.LOOK_LEFT else "right"
                        valid_faces = [fa for f, fa, ey in good_frames if fa]
                        liveness_ok = len(valid_faces) >= 3 and detect_head_turn(valid_faces, direction)[0]
                    
                    if not liveness_ok:
                        state.last_result = "LIVENESS FAILED!"
                        state.last_confidence = 0
                        state.result_time = curr_time
                        print("✗ Liveness check failed")
                    else:
                        mid_idx = len(good_frames) // 2
                        if mid_idx < len(good_frames) and state.recognizer:
                            f_mid, fa_mid, ey_mid = good_frames[mid_idx]
                            aligned, _ = extract_aligned_face(f_mid, fa_mid)
                            
                            is_match, label, confidence = recognize(state.recognizer, aligned)
                            state.last_confidence = confidence
                            
                            if is_match:
                                state.last_result = "AUTHENTICATED!"
                                print(f"✓ Authenticated (confidence: {confidence:.1f} < {Config.LBPH_THRESHOLD})")
                            else:
                                state.last_result = "ACCESS DENIED!"
                                print(f"✗ Rejected (confidence: {confidence:.1f} >= {Config.LBPH_THRESHOLD})")
                        else:
                            state.last_result = "No face / not enrolled"
                            state.last_confidence = 0
                        state.result_time = curr_time
                else:
                    state.last_result = f"Not enough good frames ({len(good_frames)}/5)"
                    state.last_confidence = 0
                    state.result_time = curr_time
                    # print(f"✗ Not enough good frames: {len(good_frames)}/5 (total: {len(state.auth_frames)})")
                
                state.mode = "IDLE"
                state.auth_frames = []
                state.auth_faces = []
                state.auth_eyes = []
        
        # Draw overlay
        frame = draw_overlay(frame, state, face, distance, quality_ok, fps)
        
        # Challenge instruction during auth
        if state.mode == "AUTHENTICATING":
            h = frame.shape[0]
            instructions = {
                Challenge.BLINK: "Please BLINK slowly",
                Challenge.LOOK_LEFT: "Turn head LEFT slowly",
                Challenge.LOOK_RIGHT: "Turn head RIGHT slowly"
            }
            instr = instructions.get(state.challenge, "")
            cv2.putText(frame, instr, (frame.shape[1]//2 - 120, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.imshow("CyberGaze Demo", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('e'):
            if state.mode == "IDLE":
                state.mode = "ENROLLING"
                state.reset_enrollment()
                print("\nStarting enrollment...")
                print("  Follow the on-screen prompts for each pose")
        
        elif key == ord('r'):
            # Re-enroll
            delete_enrolled_user()
            state.enrolled = False
            state.recognizer = None
            state.mode = "ENROLLING"
            state.reset_enrollment()
            state.last_result = "Re-enrolling..."
            state.result_time = curr_time
            print("\nRe-enrolling... Follow the prompts")
        
        elif key == ord('a'):
            if state.mode == "IDLE":
                if state.enrolled:
                    state.mode = "AUTHENTICATING"
                    state.auth_frames = []
                    state.auth_faces = []
                    state.auth_eyes = []
                    state.auth_start_time = curr_time
                    print(f"\nAuthenticating... Challenge: {state.challenge.value}")
                else:
                    state.last_result = "Enroll first (press E)"
                    state.result_time = curr_time
        
        elif key == ord('c'):
            challenge_idx = (challenge_idx + 1) % len(challenges)
            state.challenge = challenges[challenge_idx]
            print(f"Challenge: {state.challenge.value}")
        
        elif key == ord('d'):
            state.distance_gating = not state.distance_gating
            status = "ON" if state.distance_gating else "OFF"
            state.last_result = f"Distance gating: {status}"
            state.result_time = curr_time
            print(f"Distance gating: {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()