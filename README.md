# CyberGaze - Face Recognition Authentication

Advanced face recognition system with liveness detection and secure folder protection.

## Features

- **Face Recognition** - LBPH-based matching with multi-pose enrollment
- **Liveness Detection** - Blink, head turn, and texture analysis to prevent spoofing
- **Distance Validation** - Camera calibration for accurate distance gating
- **Folder Protection** - Encrypt and lock folders with biometric authentication
- **Rate Limiting** - Anti-brute-force with attempt lockout
- **Audit Logging** - Complete event logging for security

## Quick Start

### Desktop Demo
```bash
pip install -r requirements.txt
python demo.py
```
**Keys**: E=Enroll, A=Authenticate, C=Challenge, D=Distance, Q=Quit

### Flask API
```bash
python app.py
# Server at http://127.0.0.1:5000
```

## API Endpoints

- `POST /api/enroll` - Register user
- `POST /api/authenticate` - Authenticate with liveness
- `GET /api/challenge` - Get liveness challenge
- `POST /api/calibrate` - Camera calibration
- `POST /api/folder/lock` - Secure folder
- `POST /api/folder/unlock` - Unlock folder
- `GET /api/logs` - Audit logs

## Configuration

[config.py](config.py) settings:
- `LBPH_THRESHOLD = 55` - Recognition strictness (lower = stricter)
- `Z_MIN/Z_MAX = 0.35/0.80` - Distance range (meters)
- `MIN_BRIGHTNESS/MAX_BRIGHTNESS` - Quality thresholds
- `DEMO_MODE = True` - Use fixed key (set False for production)

## Modules

- **detection.py** - Face & eye detection
- **recognition.py** - LBPH training & matching
- **alignment.py** - Eye-based face alignment
- **liveness.py** - Challenge-based liveness verification
- **calibration.py** - Camera calibration & distance estimation
- **security.py** - Encryption, rate limiting, logging
- **preprocessing.py** - Image quality & normalization

## Dependencies

```
flask>=2.3.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
cryptography>=41.0.0
```

## Project Structure

```
cascades/           # Haar cascade classifiers
data/templates/     # User recognition models
modules/            # Core modules
app.py              # Flask API
demo.py             # OpenCV GUI demo
config.py           # Configuration
```
