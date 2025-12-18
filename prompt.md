# Face-Based Folder Locking Desktop Application (Electron + Python)

## Project Context

I am building a **desktop application for face-based folder access control** as a semester project.

The system must:

* Use **Electron.js** for the desktop UI
* Use **Python** as the backend for face recognition
* Allow **face enrollment**
* Allow users to **lock and unlock folders**
* Require **face verification before accessing a locked folder**
* Use a **pretrained face recognition model initially**, but be **easily replaceable with a custom model later**

The project is **NOT meant to replace OS-level authentication**, but to demonstrate **biometric access control at application level**.

---

## Functional Requirements

### 1. Face Enrollment

* Capture face using webcam
* Extract face embedding using a pretrained model
* Store embedding securely (local file or SQLite)
* Associate embedding with a user ID

### 2. Folder Management

* Allow user to:

  * Select folders using a file picker
  * Lock or unlock selected folders
* Maintain metadata:

  * Folder path
  * Lock status
  * Owner user ID

### 3. Folder Locking Logic (Windows / Linux compatible)

* When folder is locked:

  * Remove access permissions OR
  * Rename/move folder OR
  * Block opening via controlled access
* Folder must not be accessible without verification

### 4. Face Verification

* When user tries to open a locked folder:

  * Show popup window
  * Capture live face
  * Compare with enrolled embedding
  * If match → unlock folder temporarily and open it
  * If no match → deny access

### 5. Backend API

Python backend must expose REST APIs for:

* `/enroll`
* `/verify`
* `/lock-folder`
* `/unlock-folder`
* `/get-folders`

Electron must communicate with Python via **HTTP (Flask/FastAPI)**.

---

## 🏗️ System Architecture

```
Electron App (Frontend)
│
├── Webcam UI
├── Folder Selector
├── Lock / Unlock Controls
├── Verification Popup
│
└── HTTP Requests
      ↓
Python Backend (Flask/FastAPI)
│
├── Face Recognition Module
│     ├── Pretrained model (FaceNet / ArcFace)
│     ├── Embedding extraction
│     └── Cosine similarity
│
├── Folder Access Controller
│     ├── Lock / Unlock logic
│     └── OS permission handling
│
└── Local Storage
      ├── embeddings.db / JSON
      └── folder_metadata.db
```

---

## 🧩 Technical Constraints

* Face model must be **modular and replaceable**
* Backend must **not depend on Electron**
* No cloud dependency
* Must run completely offline
---

## Technology Stack

### Frontend (Electron)

### Backend (Python)
* Python 3.9+
* Flask
* OpenCV
* Pretrained Face Recognition Model:
* NumPy
* SQLite / JSON