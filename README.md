# CyberGaze — Biometric Folder Protection System

> **Technical Reference for Computer Security Semester Project**
>
> CyberGaze is a **face-recognition-based folder locking desktop application** that demonstrates multiple applied computer security concepts including biometric authentication, symmetric authenticated encryption, liveness detection, anti-spoofing, and brute-force protection — all implemented at the application level without relying on OS-level security primitives.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Complete Project Structure](#3-complete-project-structure)
4. [Security Module Deep-Dives](#4-security-module-deep-dives)
   - 4.1 [Face Recognition — `face_module.py`](#41-face-recognition--face_modulepy)
   - 4.2 [Encryption Engine — `encryption_engine.py`](#42-encryption-engine--encryption_enginepy)
   - 4.3 [Liveness Detection — `liveness_module.py`](#43-liveness-detection--liveness_modulepy)
   - 4.4 [Anti-Spoofing — `anti_spoof.py`](#44-anti-spoofing--anti_spoofpy)
   - 4.5 [Audit Logger & Brute-Force Protection — `audit_logger.py`](#45-audit-logger--brute-force-protection--audit_loggerpy)
   - 4.6 [Database Layer — `database.py`](#46-database-layer--databasepy)
   - 4.7 [Folder Controller — `folder_controller.py`](#47-folder-controller--folder_controllerpy)
5. [Flask REST API — `app.py`](#5-flask-rest-api--apppy)
6. [Complete Verification Security Pipeline](#6-complete-verification-security-pipeline)
7. [Encryption File Format](#7-encryption-file-format)
8. [Electron Security Configuration](#8-electron-security-configuration)
9. [Technology Stack](#9-technology-stack)
10. [Computer Security Concepts Demonstrated](#10-computer-security-concepts-demonstrated)
11. [Threat Model & Known Limitations](#11-threat-model--known-limitations)
12. [Installation & Usage](#12-installation--usage)
13. [Frontend Overview](#13-frontend-overview-brief)

---

## 1. Project Overview

**CyberGaze** implements **biometric access control at the application layer**. Users enroll their face once; subsequently, locked folders can only be decrypted and restored by the same face. The system is designed to demonstrate a realistic, layered security architecture using only open-source tools and local processing — no cloud, no passwords, no keys the user must remember.

### Core Security Claim

> *A folder locked by CyberGaze is protected by AES-256-GCM encryption whose key is derived deterministically from the owner's unique facial embedding. Without the owner's face passing liveness detection, anti-spoofing checks, and cosine-similarity matching, the decryption key cannot be reconstructed.*

### Scope & Threat Model Summary

- **In scope**: Casual and semi-technical attackers who do not have physical access to memory or disk forensics tools.
- **Out of scope**: OS-level attacks, kernel exploits, cold-boot attacks, or direct SQLite database manipulation.
- **Not a replacement** for OS-level disk encryption (BitLocker, FileVault, LUKS).

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ELECTRON DESKTOP APP                          │
│                                                                      │
│  ┌─────────────────┐   contextBridge    ┌────────────────────────┐  │
│  │  Renderer Process│◄──────────────────►│    Preload Script      │  │
│  │  (renderer.js)   │   (sandboxed)      │    (preload.js)        │  │
│  │  index.html      │                    │  exposeInMainWorld()   │  │
│  │  verification.html│                   └────────────┬───────────┘  │
│  └─────────────────┘                                 │IPC            │
│                                                       ▼              │
│                                          ┌────────────────────────┐  │
│                                          │   Main Process         │  │
│                                          │   (index.js)           │  │
│                                          │   BrowserWindow +      │  │
│                                          │   Electron Fuses       │  │
│                                          └────────────┬───────────┘  │
└───────────────────────────────────────────────────────┼──────────────┘
                                                        │ HTTP (localhost:5000)
                                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PYTHON FLASK BACKEND                             │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐   │
│  │   app.py     │──►│ face_module  │──►│  FaceNet (PyTorch)     │   │
│  │  REST API    │   │ .py          │   │  MTCNN detector        │   │
│  │  (719 lines) │   │ (277 lines)  │   │  Cosine similarity     │   │
│  │              │   └──────────────┘   └────────────────────────┘   │
│  │              │   ┌──────────────┐   ┌────────────────────────┐   │
│  │              │──►│ encryption   │──►│  AES-256-GCM           │   │
│  │              │   │ _engine.py   │   │  SHA-256 key derivation│   │
│  │              │   │ (283 lines)  │   │  Secure delete (3-pass)│   │
│  │              │   └──────────────┘   └────────────────────────┘   │
│  │              │   ┌──────────────┐   ┌────────────────────────┐   │
│  │              │──►│ liveness     │──►│  dlib 68-pt landmarks  │   │
│  │              │   │ _module.py   │   │  EAR blink detection   │   │
│  │              │   │ (412 lines)  │   │  solvePnP head pose    │   │
│  │              │   └──────────────┘   └────────────────────────┘   │
│  │              │   ┌──────────────┐   ┌────────────────────────┐   │
│  │              │──►│ anti_spoof   │──►│  LBP texture analysis  │   │
│  │              │   │ .py          │   │  High-freq energy      │   │
│  │              │   │ (201 lines)  │   │  Color space analysis  │   │
│  │              │   └──────────────┘   └────────────────────────┘   │
│  │              │   ┌──────────────┐   ┌────────────────────────┐   │
│  │              │──►│ audit_logger │──►│  Append-only JSON log  │   │
│  │              │   │ .py          │   │  Brute-force detection │   │
│  │              │   │ (102+ lines) │   │  Sliding window        │   │
│  │              │   └──────────────┘   └────────────────────────┘   │
│  │              │   ┌──────────────┐   ┌────────────────────────┐   │
│  │              │──►│ database.py  │──►│  SQLite                │   │
│  │              │   │ (124+ lines) │   │  Embeddings (JSON)     │   │
│  │              │   └──────────────┘   │  Folder metadata       │   │
│  │              │   ┌──────────────┐   └────────────────────────┘   │
│  │              │──►│ folder_      │──►  Rename + hidden flag        │
│  │              │   │ controller.py│    Win32 API / POSIX            │
│  │              │   │ (235 lines)  │                                 │
│  └──────────────┘   └──────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Project Structure

```
CyberGaze/
├── backend/
│   ├── app.py                 # Flask REST API server (719 lines) — central orchestrator
│   ├── face_module.py         # FaceNet face recognition module (277 lines)
│   ├── encryption_engine.py   # AES-256-GCM encryption engine (283 lines)
│   ├── liveness_module.py     # Liveness detection with challenge-response (412 lines)
│   ├── anti_spoof.py          # Anti-spoofing via LBP texture analysis (201 lines)
│   ├── audit_logger.py        # Audit logging + brute-force detection (102+ lines)
│   ├── database.py            # SQLite database for embeddings & folder metadata (124+ lines)
│   ├── folder_controller.py   # Folder lock/unlock via file system operations (235 lines)
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── index.js               # Electron main process (188 lines)
│   ├── preload.js             # Secure IPC bridge with contextIsolation (32 lines)
│   ├── renderer.js            # Frontend logic (649+ lines)
│   ├── index.html             # Main application UI
│   ├── index.css              # Application styles
│   └── verification.html      # Face verification popup
├── package.json
├── forge.config.js            # Electron Forge config with security fuses
└── prompt.md                  # Original project requirements
```

---

## 4. Security Module Deep-Dives

### 4.1 Face Recognition — `face_module.py`

**Purpose**: Extract a unique, fixed-length numerical representation (embedding) of a face that can be stored, compared, and used as a cryptographic key seed.

#### Model Architecture

| Component | Details |
|-----------|---------|
| Detector | **MTCNN** (Multi-Task Cascaded Convolutional Networks) |
| Recognizer | **InceptionResnetV1** pretrained on **VGGFace2** (via `facenet-pytorch`) |
| Input size | 160×160 pixels, margin=20 |
| Output | 512-dimensional face embedding vector |
| Hardware | CUDA GPU (auto-detected), fallback to CPU |
| Fallback | OpenCV Haar Cascade if PyTorch/FaceNet unavailable |

#### Embedding Comparison — Cosine Similarity

```
              emb_A · emb_B
similarity = ───────────────────────────────────
             ‖emb_A‖ · ‖emb_B‖

# Implementation
emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)   # normalize
similarity = np.dot(emb1_norm, emb2_norm)                      # dot product in unit sphere
similarity = (similarity + 1) / 2                              # normalize [-1,1] → [0,1] for consistent threshold semantics
```

- **Default match threshold**: `0.7` (configurable); threshold is applied after the [0,1] normalization
- Values above threshold → same person; below → different person
- Epsilon (`1e-10`) guards against division-by-zero on zero vectors

#### Image Decoding Pipeline

```
Base64 string
    → bytes (base64.b64decode)
    → PIL Image (Image.open(BytesIO))
    → RGB NumPy array
    → BGR NumPy array (OpenCV convention)
    → MTCNN detection → face crop (160×160)
    → InceptionResnetV1 inference
    → 512-dim embedding tensor
```

#### Design Decisions

- **Singleton pattern** ensures the model is loaded once per server lifetime (large PyTorch model).
- **Modular class design** (`FaceRecognition`) allows swapping the recognition backend without changing the API contract.

---

### 4.2 Encryption Engine — `encryption_engine.py`

**Purpose**: Protect the contents of locked folders using authenticated symmetric encryption derived directly from the user's biometric identity.

#### Algorithm: AES-256-GCM

AES-256-GCM (Galois/Counter Mode) is an **Authenticated Encryption with Associated Data (AEAD)** scheme providing:

| Property | Mechanism |
|----------|-----------|
| Confidentiality | AES-256 in CTR mode |
| Integrity | GHASH authentication tag (128 bits) |
| Replay protection | Unique nonce per file |
| Key size | 256 bits (32 bytes) |
| Nonce size | 96 bits (12 bytes) — NIST recommended |
| Tag size | 128 bits (16 bytes) |

#### Biometric Key Derivation

```
Face Embedding (512-dim float32 array)
    → .tobytes()          # deterministic binary serialization
    → SHA-256(bytes)      # 32-byte digest
    → AES-256 key
```

The same face will always produce the same embedding (within matching threshold), and therefore the same key — enabling passwordless decryption. No key material is stored anywhere.

> **Security Note**: This uses plain SHA-256 with no salt or iterations, not a memory-hard KDF (e.g., PBKDF2, Argon2). Biometric embeddings are continuous-valued vectors and were not designed to serve as cryptographic key material — they lack the uniform randomness ideal for key derivation. If the raw embedding is ever leaked (e.g., from an unencrypted database), an attacker can reconstruct the AES key offline at negligible cost. See [Threat Model](#11-threat-model--known-limitations) for full discussion.

#### Encrypted File Format

```
┌─────────────────────────────────────────────────────────────┐
│                   ENCRYPTED FILE LAYOUT                      │
├──────────────┬────────────┬────────────┬────────────────────┤
│ MAGIC HEADER │   NONCE    │    TAG     │    CIPHERTEXT      │
│ "CYBERGAZE"  │  12 bytes  │  16 bytes  │  (variable length) │
│  (9 bytes)   │ os.urandom │ GCM auth   │  AES-CTR encrypted │
├──────────────┴────────────┴────────────┴────────────────────┤
│ File extension: <original_filename>.cyg                      │
│ Secure container (full folder): <folder_name>.vault          │
└─────────────────────────────────────────────────────────────┘
```

- **Magic header** `"CYBERGAZE"` allows detection of already-encrypted files (prevents double encryption).
- **Unique nonce per file** via `os.urandom(12)` — critical for GCM security (nonce reuse breaks confidentiality and integrity).
- **GCM tag** is stored alongside ciphertext; decryption fails immediately and loudly if ciphertext or tag has been tampered with.

#### Secure File Deletion

Before removing original plaintext files, the engine overwrites them with random bytes in **3 passes**:

```python
for _ in range(3):
    f.write(os.urandom(file_size))
```

This mitigates recovery via file-system forensics tools (e.g., Recuva, PhotoRec) on traditional HDDs with simple file systems. It is **not** reliable on: modern SSDs (wear-leveling means overwrites may land on different physical cells), journaling file systems (ext4, NTFS — the journal may retain plaintext copies), or any copy-on-write file system (Btrfs, ZFS, APFS).

#### Secure Container

For full folder archival, the engine produces a `.vault` file:
```
Folder → tar.gz (in memory) → AES-256-GCM encryption → <folder_name>.vault
```

---

### 4.3 Liveness Detection — `liveness_module.py`

**Purpose**: Prevent **replay attacks** and **presentation attacks** where an attacker holds up a static photograph or plays a video recording of the enrolled user's face.

#### Mechanism: Challenge-Response Protocol

```
Server                               Client (webcam)
  │                                       │
  │──── GET /get-challenge ──────────────►│
  │◄─── {challenge_id, type, expires} ────│
  │                                       │
  │     [User performs action ~10s]       │
  │                                       │
  │◄─── POST /verify-liveness ────────────│
  │     {challenge_id, frames[]}          │
  │                                       │
  │──── result (pass/fail) ──────────────►│
```

- Challenges have a **60-second expiry** tracked by UUID.
- Minimum **5 frames** required per verification.
- Challenge type is randomized; client cannot pre-record the correct motion.

#### Three Challenge Types

**1. `blink_twice` — Eye Aspect Ratio (EAR)**

Uses dlib's 68 facial landmarks. Left eye: landmarks 36–41; Right eye: landmarks 42–47.

```
         ‖p2 - p6‖ + ‖p3 - p5‖
EAR = ─────────────────────────────
              2 · ‖p1 - p4‖

         p2  p3
        /      \
p1 ────          ──── p4
        \      /
         p6  p5

Threshold: EAR < 0.25  →  eyes closed
Transition: closed → open  =  one blink counted
Requirement: 2 blinks within the frame sequence
```

**2. `turn_left` — Head Pose Estimation via solvePnP**

6 landmark points (nose tip, chin, left/right eye corners, left/right mouth corners) are mapped to a 3D reference model. OpenCV's `solvePnP` solves for rotation and translation vectors. The rotation vector is converted to a rotation matrix via Rodrigues' formula, then decomposed into Euler angles.

```
Rotation Vector (rvec)
    → cv2.Rodrigues()  →  3×3 Rotation Matrix
    → decomposeProjectionMatrix()
    → Euler angles: [pitch, yaw, roll]

Yaw < -15°  →  head turned left  ✓
Yaw > +15°  →  head turned right ✓
```

**3. `turn_right`** — Same as above with yaw > +15°.

#### Model Dependency

- **dlib 68-point facial landmark model**: `shape_predictor_68_face_landmarks.dat` (~100 MB), auto-downloaded on first run.
- Graceful fallback to a simplified detector if dlib is unavailable.

---

### 4.4 Anti-Spoofing — `anti_spoof.py`

**Purpose**: Detect **Presentation Attacks** — attempts to fool the face recognizer using a printed photograph or a screen displaying the enrolled user's face.

**Key Design**: No deep neural network required — three lightweight classical computer vision features are combined with a weighted scoring function.

#### Three-Feature Scoring System

**Feature 1: LBP Histogram Analysis (weight: 0.40)**

Local Binary Pattern (LBP) encodes micro-texture by comparing each pixel to its 8 neighbors (radius=1):

```
For each pixel P at position (x, y):
  For each neighbor n_i at radius r, angle θ_i:
    bit_i = 1 if n_i >= P, else 0
  LBP_value = Σ bit_i · 2^i    (8-bit code, 0–255)

Build histogram of LBP values over ROI.
Compute entropy and variance of histogram.
```

- **Real faces**: natural skin micro-texture produces **high entropy** LBP histograms with many distinct patterns.
- **Printed/screen images**: ink dot patterns, moiré effects, and pixel grids produce **low entropy** histograms with pronounced peaks.

**Feature 2: High-Frequency Energy (weight: 0.35)**

```
Laplacian variance = Var( ∇²I )

Real face:  natural skin pores, hair, fine lines → high HF energy
Printed:    ink diffusion, screen pixel blur     → low HF energy

Normalized: score = clamp((variance - 10) / (1000 - 10), 0, 1)
Threshold:  HIGH_FREQ_THRESHOLD = 0.15
```

**Feature 3: Color Space Analysis (weight: 0.25)**

```
BGR image → LAB color space
Analyze chrominance channels: a* (green–red), b* (blue–yellow)
Metric: standard deviation of a and b channels

Real face: under natural illumination, skin has characteristic
           chrominance variance due to subsurface scattering.
Printed:   different chrominance distribution; ink color gamut differs.

Normalized: score = clamp((std_dev - 5) / (30 - 5), 0, 1)
```

#### Combined Score

```
spoof_score = 0.40 × lbp_score + 0.35 × hf_score + 0.25 × color_score

SPOOF_THRESHOLD = 0.4
score ≥ 0.4  →  REAL face  ✓
score < 0.4  →  SPOOF detected  ✗  (logged as SPOOF_ATTEMPT)
```

---

### 4.5 Audit Logger & Brute-Force Protection — `audit_logger.py`

**Purpose**: Maintain a tamper-evident record of all security-relevant events and automatically lock out accounts under brute-force attack.

#### Append-Only Audit Log

Log file: `cybergaze.audit.log`

Each entry is a JSON object containing:

```json
{
  "timestamp": "2025-01-15T14:32:01.123456",
  "epoch": 1736950321.123456,
  "event_type": "VERIFY",
  "user_id": "alice",
  "folder_id": "documents_2025",
  "success": false,
  "details": { "reason": "similarity_below_threshold", "score": 0.61 }
}
```

| Event Type | Trigger |
|------------|---------|
| `ENROLL` | Face enrollment attempt |
| `VERIFY` | Face verification attempt |
| `LOCK` | Folder lock operation |
| `UNLOCK` | Folder unlock operation |
| `SPOOF_ATTEMPT` | Anti-spoof check failed |
| `LIVENESS_FAIL` | Liveness challenge not completed |
| `LOCKOUT` | Account locked due to brute-force |
| `ACCOUNT_UNLOCKED` | Lockout period expired |

#### Brute-Force Detection — Sliding Window Algorithm

```
Constants:
  MAX_FAILURES     = 3        # failures before lockout
  FAILURE_WINDOW   = 300 s    # 5-minute sliding window
  LOCKOUT_DURATION = 600 s    # 10-minute lockout

For each incoming verification request:
  1. Clean up failure records older than FAILURE_WINDOW
  2. Count remaining failures for user_id
  3. If count >= MAX_FAILURES → reject with LOCKOUT event
  4. If verification fails → record timestamp in memory
  5. If failures reach MAX_FAILURES → log LOCKOUT event

Tracked events: VERIFY, SPOOF_ATTEMPT, LIVENESS_FAIL
```

Failure data is stored **in-memory** (dict keyed by user_id), providing fast O(1) lookup. Limitation: resets on server restart. All three constants (`MAX_FAILURES`, `FAILURE_WINDOW`, `LOCKOUT_DURATION`) are hardcoded module-level values in `audit_logger.py`; there is no runtime configuration file for them.

#### Threat Analysis Endpoint

`GET /threat-analysis` returns:
- Total events by type
- Users with elevated failure rates
- Recent SPOOF_ATTEMPT events
- Currently locked-out accounts

---

### 4.6 Database Layer — `database.py`

**Storage**: Local **SQLite** database (`cybergaze.db`), kept entirely on the user's machine.

#### Schema

```sql
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    UNIQUE NOT NULL,
    embedding   TEXT    NOT NULL,   -- JSON-serialized numpy float32 array
    status      TEXT    DEFAULT 'active',
    created_at  TEXT,
    updated_at  TEXT
);

CREATE TABLE folders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id        TEXT    UNIQUE NOT NULL,
    original_path    TEXT    NOT NULL,
    is_locked        INTEGER DEFAULT 0,
    is_encrypted     INTEGER DEFAULT 0,
    encryption_salt  TEXT,
    owner_id         TEXT    REFERENCES users(user_id),
    created_at       TEXT
);
```

- Face embeddings are stored as **JSON-serialized NumPy arrays** (`list(embedding.numpy())`).
- The `owner_id` foreign key implements the **ownership model** — only the enrolling user can unlock their folders.
- Database migration support via `ALTER TABLE` for backwards compatibility when new columns are added.

---

### 4.7 Folder Controller — `folder_controller.py`

**Purpose**: Implement the physical access control mechanism — making folders inaccessible without decryption.

#### Lock Sequence

```
1. validate(folder_path)
      └─ folder exists? not already locked? owner matches?
2. encrypt(folder_path, aes_key)
      └─ recursively encrypt all files → .cyg extensions
         original files securely deleted (3-pass overwrite)
3. rename(folder_path → .cybergaze_locked_<original_name>)
4. [Windows only] SetFileAttributes(FILE_ATTRIBUTE_HIDDEN)
      └─ ctypes.windll.kernel32.SetFileAttributesW()
```

#### Unlock Sequence

```
1. validate(locked_folder_path)
      └─ locked folder exists? requestor is owner?
2. rename(.cybergaze_locked_<name> → <original_name>)
3. decrypt(folder_path, aes_key)
      └─ recursively decrypt all .cyg files
         verify GCM tag before accepting plaintext
```

#### Cross-Platform Notes

| Platform | Hide Mechanism |
|----------|----------------|
| Windows  | `FILE_ATTRIBUTE_HIDDEN` via Win32 API (ctypes) |
| macOS    | Folder renamed with `.` prefix (POSIX hidden) |
| Linux    | Folder renamed with `.` prefix (POSIX hidden) |

---

## 5. Flask REST API — `app.py`

Central orchestrator (719 lines) connecting all security modules.

| Endpoint | Method | Security Features |
|----------|--------|-------------------|
| `/health` | GET | Service status + module availability check |
| `/enroll` | POST | Face enrollment with audit logging |
| `/verify` | POST | Anti-spoof → liveness check → cosine similarity → lockout check → audit |
| `/lock-folder` | POST | Enrollment check + embedding → AES key derivation + audit |
| `/unlock-folder` | POST | Requires `verified: true` + ownership check + AES decryption + audit |
| `/get-folders` | GET | Lists tracked folders with lock/encryption status |
| `/add-folder` | POST | Adds folder to tracking database |
| `/remove-folder` | POST | Removes tracking (folder must be unlocked first) |
| `/users` | GET | List all enrolled users |
| `/users/<id>` | DELETE | Delete user and associated data |
| `/users/<id>/exists` | GET | Check enrollment status |
| `/get-challenge` | GET | Generate UUID-tagged liveness challenge |
| `/verify-liveness` | POST | Verify challenge with multi-frame landmark analysis |
| `/security-status` | GET | Module availability + real-time threat analysis |
| `/audit-logs` | GET | Paginated, filterable audit log retrieval |
| `/threat-analysis` | GET | Structured threat analysis report |

### Key Security Logic in `/verify`

```
POST /verify
  ├── 1. Check lockout status (audit_logger)
  │        └─ REJECT if locked out
  ├── 2. Anti-spoof check (anti_spoof)
  │        └─ REJECT + log SPOOF_ATTEMPT if score < 0.4
  ├── 3. Extract face embedding (face_module)
  │        └─ REJECT if no face detected
  ├── 4. Load stored embedding from DB (database)
  │        └─ REJECT if user not enrolled
  ├── 5. Cosine similarity comparison
  │        └─ REJECT + log VERIFY(fail) if similarity < 0.7
  └── 6. Return verified=true + embedding for key derivation
```

---

## 6. Complete Verification Security Pipeline

The end-to-end flow for unlocking a folder, showing every security gate:

```
User clicks "Unlock"
        │
        ▼
┌───────────────────┐
│  1. Camera Access │  ← Browser WebRTC API, user must grant permission
└────────┬──────────┘
         │
         ▼
┌──────────────────────────┐
│  2. Get Liveness         │  GET /get-challenge
│     Challenge            │  ← Server generates random challenge (blink/turn_left/turn_right)
│                          │     with UUID + 60s expiry
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  3. Capture Frames       │  ← min 5 frames at ~200ms intervals
│                          │     JPEG quality 0.7, Base64 encoded
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  4. Verify Liveness      │  POST /verify-liveness
│  [SECURITY GATE 1]       │  ← dlib landmark analysis on each frame
│                          │     EAR blink counting OR solvePnP head pose
│                          │     FAIL → log LIVENESS_FAIL, reject
└────────┬─────────────────┘
         │  liveness passed
         ▼
┌──────────────────────────┐
│  5. Anti-Spoof Check     │  POST /verify (middle frame)
│  [SECURITY GATE 2]       │  ← LBP texture + HF energy + color space
│                          │     score < 0.4 → log SPOOF_ATTEMPT, reject
└────────┬─────────────────┘
         │  real face confirmed
         ▼
┌──────────────────────────┐
│  6. Face Identity Match  │
│  [SECURITY GATE 3]       │  ← FaceNet embedding extraction
│                          │     cosine similarity vs stored embedding
│                          │     similarity < 0.7 → log VERIFY(fail)
│                          │     >= 3 failures → LOCKOUT (10 min)
└────────┬─────────────────┘
         │  identity confirmed, embedding returned
         ▼
┌──────────────────────────┐
│  7. Key Derivation       │  ← SHA-256( embedding.tobytes() ) → 32-byte AES key
│                          │     No key stored anywhere; key exists only in memory
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  8. AES-256-GCM          │  POST /unlock-folder
│     Decryption           │  ← Decrypt each .cyg file
│  [SECURITY GATE 4]       │     Verify GCM tag → reject if tampered
│                          │     Restore original filenames
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  9. Folder Restoration   │  ← Rename .cybergaze_locked_X → X
│                          │     Remove hidden attribute (Windows)
│                          │     Log UNLOCK(success) to audit log
└──────────────────────────┘
         │
         ▼
    Folder accessible ✓
```

---

## 7. Encryption File Format

### Per-File Encrypted Format (`.cyg`)

```
Byte offset:   0         9        21       37       37+N
               │         │         │         │         │
               ▼         ▼         ▼         ▼         ▼
          ┌─────────┬────────────┬──────────┬──────────────┐
          │ MAGIC   │   NONCE    │   TAG    │  CIPHERTEXT  │
          │"CYBERGAZE│  12 bytes  │ 16 bytes │  N bytes     │
          │ 9 bytes │ os.urandom │ GCM auth │ AES-CTR enc  │
          └─────────┴────────────┴──────────┴──────────────┘
```

### Secure Container Format (`.vault`)

```
Original Folder
    │
    ▼ tarfile (in memory, gzip compressed)
    │
    ▼ AES-256-GCM encryption (same file layout as .cyg)
    │
    ▼  <folder_name>.vault
```

### Key Derivation Flow

```
┌──────────────────────────────────────┐
│  512-dim float32 embedding vector    │
│  (from FaceNet InceptionResnetV1)    │
└──────────────┬───────────────────────┘
               │ .tobytes() → 2048 bytes
               ▼
┌──────────────────────────────────────┐
│  SHA-256                             │
│  Input:  2048-byte embedding binary  │
│  Output: 32-byte digest              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  AES-256-GCM Key (32 bytes = 256 bit)│
│  Used directly with pycryptodome     │
│  AES.new(key, AES.MODE_GCM, nonce)   │
└──────────────────────────────────────┘
```

---

## 8. Electron Security Configuration

The frontend is hardened against common Electron security pitfalls.

### Process Isolation Model

```
┌─────────────────────────────────────────────────────┐
│  Main Process (Node.js — full OS access)             │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │  Renderer Process (sandboxed)                  │ │
│  │  contextIsolation: true                        │ │
│  │  nodeIntegration: false                        │ │
│  │  sandbox: true                                 │ │
│  │                                                │ │
│  │  Can ONLY call methods exposed via:            │ │
│  │  contextBridge.exposeInMainWorld('api', {...}) │ │
│  └────────────────────────────────────────────────┘ │
│           │ IPC (ipcRenderer.invoke)                 │
│           ▼                                          │
│  preload.js  ←  whitelist of allowed operations     │
└─────────────────────────────────────────────────────┘
```

### Security Configuration (forge.config.js)

| Electron Fuse | Value | Security Purpose |
|---------------|-------|-----------------|
| `RunAsNode` | `false` | Prevents using app binary as a plain Node.js interpreter |
| `EnableCookieEncryption` | `true` | Encrypts session cookies at rest |
| `EnableNodeOptionsEnvironmentVariable` | `false` | Prevents `NODE_OPTIONS` env var injection |
| `EnableNodeCliInspectArguments` | `false` | Prevents `--inspect` debugger attachment |
| `EnableEmbeddedAsarIntegrityValidation` | `true` | Validates ASAR archive integrity on load |
| `OnlyLoadAppFromAsar` | `true` | Prevents loading app code from outside the ASAR |

### Content Security Policy (verification.html)

```http
Content-Security-Policy:
  default-src 'self';
  script-src  'self';
  connect-src 'self' http://localhost:5000;
  img-src     'self' data:;
  style-src   'self' 'unsafe-inline';
```

- Restricts backend API calls strictly to `localhost:5000`.
- Prevents injection of remote scripts or data exfiltration.

---

## 9. Technology Stack

### Backend (Python)

| Library | Version | Purpose |
|---------|---------|---------|
| Flask | 2.3+ | REST API framework |
| facenet-pytorch | 2.5+ | FaceNet/MTCNN face recognition |
| PyTorch | 2.0+ | Deep learning inference |
| OpenCV (`cv2`) | 4.8+ | Image processing, LBP, solvePnP |
| dlib | 19.24+ | 68-point facial landmark detection |
| pycryptodome | 3.19+ | AES-256-GCM encryption |
| NumPy | latest | Numerical operations, embedding math |
| Pillow (PIL) | latest | Image format conversion |
| SQLite3 | stdlib | Local database |

### Frontend (Electron)

| Component | Version | Purpose |
|-----------|---------|---------|
| Electron | 39.x | Desktop app framework |
| Electron Forge | 7.10.x | Build, packaging, fuses |
| HTML5/CSS3/JS | — | UI rendering |
| WebRTC (`getUserMedia`) | — | Webcam access for face capture |

---

## 10. Computer Security Concepts Demonstrated

This project is designed as a practical demonstration of the following Computer Security course topics:

| # | CS Security Concept | Implementation in CyberGaze |
|---|--------------------|-----------------------------|
| 1 | **Biometric Authentication** | FaceNet face embeddings as the sole authentication factor |
| 2 | **Symmetric Encryption** | AES-256-GCM for data-at-rest protection of folder contents |
| 3 | **Authenticated Encryption (AEAD)** | GCM mode provides confidentiality + integrity in one primitive |
| 4 | **Biometric Key Derivation** | Face embedding → SHA-256 → AES key (no passwords) |
| 5 | **Presentation Attack Detection (PAD)** | LBP texture analysis to reject printed/screen spoofs |
| 6 | **Liveness Detection** | Challenge-response (blink, head turn) to prevent replay attacks |
| 7 | **Replay Attack Prevention** | UUID-tagged challenges with 60s expiry, multi-frame requirement |
| 8 | **Brute-Force Protection** | Sliding-window failure tracking → automatic account lockout |
| 9 | **Audit Logging** | Tamper-evident append-only JSON log of all security events |
| 10 | **Principle of Least Privilege** | Electron contextIsolation, nodeIntegration=false, sandboxed renderer |
| 11 | **Defense in Depth** | Four sequential security gates: liveness → anti-spoof → face match → GCM integrity |
| 12 | **Access Control / Ownership Model** | Folder owner enforced via DB foreign key; only owner can unlock |
| 13 | **Secure Random Number Generation** | `os.urandom()` for GCM nonces — CSPRNG, not `random` module |
| 14 | **Secure Deletion** | 3-pass random overwrite before removing plaintext files |
| 15 | **Code Integrity** | Electron Fuses: ASAR integrity validation, load-only-from-ASAR |
| 16 | **Threat Monitoring** | Real-time threat analysis endpoint; anomaly detection on audit log |

---

## 11. Threat Model & Known Limitations

### Addressed Threats

| Threat | Mitigation |
|--------|-----------|
| Casual unauthorized access | Folder hidden + renamed (inaccessible without app) |
| Static photo spoofing | Anti-spoofing LBP texture analysis |
| Video replay spoofing | Liveness challenge-response (blink/head turn) |
| Brute-force face matching | Sliding-window lockout (3 failures / 5 min → 10 min ban) |
| File tampering | AES-256-GCM authentication tag rejects modified ciphertext |
| Plaintext recovery after encryption | 3-pass secure deletion of original files |
| Unauthorized audit log reading | Structured JSON; designed for append-only access |
| Electron code injection | contextIsolation, nodeIntegration=false, CSP |
| Electron binary tampering | ASAR integrity validation, OnlyLoadAppFromAsar |

### Known Limitations

| Limitation | Risk Level | Notes |
|------------|-----------|-------|
| Application-level, not OS-level protection | Medium | Advanced users can directly access renamed/hidden folders via terminal or file manager |
| SHA-256 key derivation (no PBKDF2/Argon2) | Medium | No key stretching; key derivation is fast, reducing offline attack cost if embedding is leaked |
| In-memory brute-force tracking | Low | Failure counters reset on server restart; persistent tracking would require DB storage |
| No database encryption | Low | `cybergaze.db` stores embeddings in plaintext JSON; an attacker with file access could extract embeddings |
| Single-factor biometric | Medium–High | No MFA; modern 3D-printed silicone masks and high-resolution video deepfakes have been demonstrated against commercial liveness systems. The LBP-based anti-spoofing and EAR/pose liveness detection are classical methods, not deep-learning PAD, increasing susceptibility to adaptive attacks. |
| Cosine similarity threshold | Low | Fixed threshold (0.7) may yield false positives/negatives depending on lighting and camera quality |
| Secure deletion limitations | Low | Does not protect against SSD wear-leveling or journaling file systems (ext4, NTFS) |
| Local SQLite, no WAL encryption | Low | Database write-ahead log may contain plaintext copies of embeddings |

### Attack Surface Summary

```
External Attack Surface:
  • Flask API on localhost:5000 (loopback only; not exposed to network)
  • No authentication on API endpoints (relies on OS-level user isolation)
    ⚠ Any process running under the same OS user account — including
      malware, browser extensions, or other apps — can call all endpoints
      (lock/unlock folders, enumerate users, read audit logs, trigger
      enrollment) without presenting any credential. A token-based or
      shared-secret authentication layer would eliminate this risk.

Internal Attack Surface:
  • SQLite database file (embeddings stored as plaintext JSON; if
    embedding is read, the AES key can be reconstructed offline via
    SHA-256 — no key stretching prevents this)
  • In-memory AES key (exists only during decryption; cleared after use)
  • Audit log file (append-only by convention, not enforced by OS)
  • ASAR package (integrity validated by Electron Fuse)
```

---

## 12. Installation & Usage

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | 18+ |
| Python | 3.9+ |
| pip | latest |
| Webcam | Required for face capture |

### 1. Clone the Repository

```bash
git clone https://github.com/Rag-795/CyberGaze.git
cd CyberGaze
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note**: First run will auto-download the dlib 68-point landmark model (~100 MB) and the FaceNet/VGGFace2 pretrained weights. Ensure you have ~2 GB free disk space and a stable internet connection for the first launch.

### 4. Start the Python Backend

```bash
cd backend
python app.py
```

Expected output:
```
==================================================
CyberGaze Backend Server
==================================================
[+] FaceNet (MTCNN + InceptionResnetV1) loaded — device: cpu
[+] dlib landmark model loaded
[+] AES-256-GCM encryption engine ready
[+] Audit logger initialized
Starting server on http://localhost:5000
```

### 5. Start the Electron Application

In a separate terminal:

```bash
npm start
```

### Usage Workflow

#### Enroll Your Face
1. Navigate to **Face Enrollment** in the sidebar.
2. Enter a unique User ID.
3. Click **Start Camera** → allow webcam access.
4. Position your face within the guide circle with good lighting.
5. Click **Capture & Enroll** — the embedding is extracted and stored.

#### Lock a Folder
1. Navigate to **Folders** → **Add Folder** → select a directory.
2. Click **Lock** on the folder entry.
3. The folder is encrypted (AES-256-GCM) and hidden.

#### Unlock a Folder
1. Click **Unlock** on a locked folder entry.
2. The verification popup opens.
3. Complete the liveness challenge (blink twice, or turn head as directed).
4. Hold your face steady — anti-spoof and face match run automatically.
5. On success: the AES key is derived in-memory, files are decrypted, folder is restored.

### API Quick Reference

```bash
# Check backend health
curl http://localhost:5000/health

# Enroll face (base64 image)
curl -X POST http://localhost:5000/enroll \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "image": "<base64>"}'

# Get liveness challenge
curl http://localhost:5000/get-challenge

# Check security status
curl http://localhost:5000/security-status

# View audit logs
curl "http://localhost:5000/audit-logs?page=1&per_page=50"

# View threat analysis
curl http://localhost:5000/threat-analysis
```

---

## 13. Frontend Overview (Brief)

The frontend is intentionally minimal — its only purpose is to provide a user interface for the security operations implemented in the backend.

- **Electron.js** (`src/index.js`, 188 lines): Manages the application window lifecycle, enforces security fuses, spawns the Python backend process as a child process.
- **Preload script** (`src/preload.js`, 32 lines): The sole communication bridge between the sandboxed renderer and the main process via `contextBridge.exposeInMainWorld`. Only explicitly whitelisted operations are available to the UI.
- **Main UI** (`src/index.html` + `src/renderer.js`, 649+ lines): Single-page application for enrollment, folder management, and audit log viewing.
- **Verification popup** (`src/verification.html`): Isolated window for the face verification flow with its own CSP header. Accesses webcam via WebRTC `getUserMedia`, performs frame capture, and calls the Flask API directly over `http://localhost:5000`.

The UI/CSS (`src/index.css`) uses standard web technologies and is not security-critical.

---

## Security Architecture — One-Line Summary

> CyberGaze implements a **four-layer biometric access control system**: liveness detection (challenge-response) → anti-spoofing (LBP texture) → face identity verification (FaceNet cosine similarity) → authenticated encryption (AES-256-GCM with biometric key derivation), logging every event to a tamper-evident audit trail with automatic brute-force lockout.

---

*CyberGaze — Computer Security Semester Project*
