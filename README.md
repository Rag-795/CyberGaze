# CyberGaze - Face-Based Folder Locking Application

A desktop application that uses face recognition to protect your folders. Built with **Electron.js** for the UI and **Python (Flask)** for the backend face recognition.

![CyberGaze Screenshot](docs/screenshot.png)

## ✨ Features

- **Face Enrollment** - Register your face using webcam capture
- **Folder Protection** - Lock/unlock folders with face verification
- **Secure Access** - Locked folders are hidden and renamed for protection
- **Offline Mode** - Works completely offline, no cloud dependency
- **Modular Design** - Easy to replace face recognition model (uses FaceNet)

## 🛠️ Technology Stack

### Frontend (Electron)
- Electron.js
- HTML5/CSS3/JavaScript
- Webcam API for face capture

### Backend (Python)
- Flask with CORS
- FaceNet via `facenet-pytorch` (PyTorch)
- MTCNN for face detection
- OpenCV for image processing
- SQLite for local database

## 📋 Prerequisites

- **Node.js** (v18 or later)
- **Python** (3.9 or later)
- **pip** (Python package manager)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CyberGaze.git
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

> **Note**: FaceNet requires TensorFlow. The first run may take a few minutes to download the model.

## 🎮 Running the Application

### Step 1: Start the Python Backend

Open a terminal and run:

```bash
cd backend
python app.py
```

You should see:
```
==================================================
CyberGaze Backend Server
==================================================
Starting server on http://localhost:5000
```

### Step 2: Start the Electron App

In a new terminal:

```bash
npm start
```

## 📖 Usage Guide

### 1. Enroll Your Face
1. Click **"Face Enrollment"** in the sidebar
2. Enter a unique User ID
3. Click **"Start Camera"** and allow webcam access
4. Position your face within the guide circle
5. Click **"Capture & Enroll"**

### 2. Add a Folder to Protect
1. Click **"Folders"** in the sidebar
2. Click **"Add Folder"** and select a folder
3. The folder is now tracked

### 3. Lock a Folder
1. Find the folder in your list
2. Click the **"Lock"** button
3. The folder will be hidden and protected

### 4. Unlock a Folder
1. Click **"Unlock"** on a locked folder
2. A verification popup will appear
3. Show your face to the camera
4. Click **"Verify & Unlock"**
5. If verified, the folder will be restored

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/enroll` | POST | Enroll face with user_id and image |
| `/verify` | POST | Verify face against enrolled user |
| `/lock-folder` | POST | Lock a folder |
| `/unlock-folder` | POST | Unlock a folder (requires verification) |
| `/get-folders` | GET | List all protected folders |
| `/add-folder` | POST | Add folder to tracking |
| `/remove-folder` | POST | Remove folder from tracking |

## 🏗️ Project Structure

```
CyberGaze/
├── backend/
│   ├── app.py              # Flask API server
│   ├── database.py         # SQLite database operations
│   ├── face_module.py      # FaceNet face recognition
│   ├── folder_controller.py # Folder lock/unlock logic
│   └── requirements.txt    # Python dependencies
├── src/
│   ├── index.js           # Electron main process
│   ├── preload.js         # Secure IPC bridge
│   ├── index.html         # Main application UI
│   ├── index.css          # Application styles
│   ├── renderer.js        # Frontend logic
│   └── verification.html  # Face verification popup
├── package.json
└── README.md
```

## 🔐 How Folder Locking Works

1. When a folder is locked, it is renamed with a `.cybergaze_locked_` prefix
2. On Windows, the hidden attribute is also set
3. The folder becomes invisible to casual browsing
4. To access, face verification must pass
5. Upon successful verification, the folder is restored to its original name

> **Note**: This is application-level protection for demonstration purposes, not OS-level encryption.

## 🎨 Customization

### Replacing the Face Recognition Model

The face recognition module (`backend/face_module.py`) is designed to be modular. To use a different model:

1. Modify the `FaceRecognition` class
2. Update `extract_embedding()` to use your model
3. Adjust the similarity threshold as needed

### Adjusting Verification Threshold

In `face_module.py`, modify the threshold:

```python
_face_recognition_instance = FaceRecognition(threshold=0.6)  # Default: 0.6
```

- Higher = stricter matching
- Lower = more lenient matching

## 🐛 Troubleshooting

### Backend won't start
- Ensure Python 3.9+ is installed
- Check if port 5000 is available
- Install missing dependencies: `pip install -r requirements.txt`

### Camera not working
- Allow camera permissions in your browser/OS
- Check if another app is using the camera

### Face not detected
- Ensure good lighting
- Face the camera directly
- Keep face within the guide circle

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**CyberGaze** - Protect your folders with your face 👁️
