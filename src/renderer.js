/**
 * CyberGaze - Renderer Process
 * Frontend application logic for face-based folder protection
 */

// ============ Configuration ============
const API_BASE_URL = 'http://localhost:5000';
let currentUserId = localStorage.getItem('cybergaze_user_id') || 'default_user';

// ============ DOM Elements ============
const elements = {
    // Views
    views: document.querySelectorAll('.view'),
    navBtns: document.querySelectorAll('.nav-btn'),

    // Dashboard
    statEnrolled: document.getElementById('stat-enrolled'),
    statFolders: document.getElementById('stat-folders'),
    statLocked: document.getElementById('stat-locked'),

    // Enrollment
    enrollVideo: document.getElementById('enroll-video'),
    enrollCanvas: document.getElementById('enroll-canvas'),
    btnStartCamera: document.getElementById('btn-start-camera'),
    btnCapture: document.getElementById('btn-capture'),
    cameraStatus: document.getElementById('camera-status'),
    userIdInput: document.getElementById('user-id-input'),
    enrollmentStatus: document.getElementById('enrollment-status'),

    // Folders
    foldersList: document.getElementById('folders-list'),
    btnAddFolder: document.getElementById('btn-add-folder'),
    btnRefreshFolders: document.getElementById('btn-refresh-folders'),

    // Toasts
    toastContainer: document.getElementById('toast-container'),

    // Connection status
    connectionStatus: document.getElementById('connection-status'),
};

// ============ State ============
let mediaStream = null;
let isEnrolled = false;
let folders = [];

// ============ Initialization ============
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initWindowControls();
    initEnrollment();
    initFolders();
    checkBackendConnection();
    loadDashboardStats();

    // Set user ID from storage
    if (elements.userIdInput) {
        elements.userIdInput.value = currentUserId;
    }

    // Listen for folder unlock events
    if (window.electronAPI) {
        window.electronAPI.onFolderUnlocked((data) => {
            showToast('success', 'Folder Unlocked', `${data.folderName} has been unlocked`);
            loadFolders();
            loadDashboardStats();
        });
    }
});

// ============ Navigation ============
function initNavigation() {
    elements.navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const viewId = btn.dataset.view;
            switchView(viewId);
        });
    });
}

function switchView(viewId) {
    // Update nav buttons
    elements.navBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewId);
    });

    // Update views
    elements.views.forEach(view => {
        view.classList.toggle('active', view.id === `view-${viewId}`);
    });

    // Stop camera when leaving enrollment view
    if (viewId !== 'enroll' && mediaStream) {
        stopCamera();
    }
}

// Make switchView available globally for onclick handlers
window.switchView = switchView;

// ============ Window Controls ============
function initWindowControls() {
    if (!window.electronAPI) return;

    document.getElementById('btn-minimize')?.addEventListener('click', () => {
        window.electronAPI.minimize();
    });

    document.getElementById('btn-maximize')?.addEventListener('click', () => {
        window.electronAPI.maximize();
    });

    document.getElementById('btn-close')?.addEventListener('click', () => {
        window.electronAPI.close();
    });
}

// ============ API Calls ============
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function checkBackendConnection() {
    try {
        const result = await apiCall('/health');
        if (result.status === 'healthy') {
            updateConnectionStatus(true);
            return true;
        }
    } catch (error) {
        updateConnectionStatus(false);
    }
    return false;
}

function updateConnectionStatus(connected) {
    const statusEl = elements.connectionStatus;
    if (!statusEl) return;

    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('span');

    if (connected) {
        dot.classList.add('connected');
        dot.classList.remove('disconnected');
        text.textContent = 'Backend Online';
    } else {
        dot.classList.remove('connected');
        dot.classList.add('disconnected');
        text.textContent = 'Backend Offline';
    }
}

// ============ Dashboard ============
async function loadDashboardStats() {
    try {
        // Check enrollment status
        const userCheck = await apiCall(`/users/${currentUserId}/exists`);
        isEnrolled = userCheck.exists;
        elements.statEnrolled.textContent = isEnrolled ? 'Yes' : 'No';

        // Load folders
        const foldersResult = await apiCall(`/get-folders?owner_id=${currentUserId}`);
        if (foldersResult.success) {
            folders = foldersResult.folders;
            elements.statFolders.textContent = folders.length;
            elements.statLocked.textContent = folders.filter(f => f.is_locked).length;
        }

        updateEnrollmentStatus();
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

// ============ Face Enrollment ============
function initEnrollment() {
    elements.btnStartCamera?.addEventListener('click', toggleCamera);
    elements.btnCapture?.addEventListener('click', captureAndEnroll);

    elements.userIdInput?.addEventListener('change', (e) => {
        currentUserId = e.target.value.trim() || 'default_user';
        localStorage.setItem('cybergaze_user_id', currentUserId);
        loadDashboardStats();
    });
}

async function toggleCamera() {
    if (mediaStream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        elements.cameraStatus.textContent = 'Starting camera...';

        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        elements.enrollVideo.srcObject = mediaStream;
        elements.btnStartCamera.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2"/>
        <path d="M23 7l-7 5 7 5V7z"/>
        <line x1="1" y1="1" x2="23" y2="23"/>
      </svg>
      Stop Camera
    `;
        elements.btnCapture.disabled = false;
        elements.cameraStatus.textContent = 'Camera active - Position your face';

    } catch (error) {
        console.error('Camera error:', error);
        elements.cameraStatus.textContent = 'Camera access denied';
        showToast('error', 'Camera Error', 'Could not access camera. Please check permissions.');
    }
}

function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    elements.enrollVideo.srcObject = null;
    elements.btnStartCamera.innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M23 7l-7 5 7 5V7z"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
    </svg>
    Start Camera
  `;
    elements.btnCapture.disabled = true;
    elements.cameraStatus.textContent = 'Camera Ready';
}

async function captureAndEnroll() {
    if (!mediaStream) {
        showToast('error', 'Error', 'Please start the camera first');
        return;
    }

    const userId = elements.userIdInput.value.trim();
    if (!userId) {
        showToast('error', 'Error', 'Please enter a User ID');
        return;
    }

    try {
        elements.btnCapture.disabled = true;
        elements.cameraStatus.textContent = 'Capturing...';

        // Capture frame from video
        const canvas = elements.enrollCanvas;
        const video = elements.enrollVideo;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        // Flip horizontally to match mirror view
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);

        // Get base64 image
        const imageData = canvas.toDataURL('image/jpeg', 0.9);

        elements.cameraStatus.textContent = 'Processing face...';

        // Send to backend
        const result = await apiCall('/enroll', {
            method: 'POST',
            body: JSON.stringify({
                user_id: userId,
                image: imageData
            })
        });

        if (result.success) {
            currentUserId = userId;
            localStorage.setItem('cybergaze_user_id', userId);
            isEnrolled = true;

            showToast('success', 'Enrollment Successful', result.message);
            updateEnrollmentStatus();
            loadDashboardStats();
            elements.cameraStatus.textContent = 'Face enrolled successfully!';
        } else {
            showToast('error', 'Enrollment Failed', result.error);
            elements.cameraStatus.textContent = result.error;
        }

    } catch (error) {
        console.error('Enrollment error:', error);
        showToast('error', 'Error', 'Failed to enroll face. Is the backend running?');
        elements.cameraStatus.textContent = 'Enrollment failed';
    } finally {
        elements.btnCapture.disabled = false;
    }
}

function updateEnrollmentStatus() {
    const statusEl = elements.enrollmentStatus;
    if (!statusEl) return;

    if (isEnrolled) {
        statusEl.innerHTML = `
      <div class="status-icon success">✓</div>
      <span>Face enrolled for user: ${currentUserId}</span>
    `;
    } else {
        statusEl.innerHTML = `
      <div class="status-icon pending">○</div>
      <span>Not enrolled yet</span>
    `;
    }
}

// ============ Folders Management ============
function initFolders() {
    elements.btnAddFolder?.addEventListener('click', addFolder);
    elements.btnRefreshFolders?.addEventListener('click', loadFolders);

    loadFolders();
}

async function loadFolders() {
    try {
        const result = await apiCall(`/get-folders?owner_id=${currentUserId}`);

        if (result.success) {
            folders = result.folders;
            renderFolders();

            // Update dashboard stats
            if (elements.statFolders) {
                elements.statFolders.textContent = folders.length;
            }
            if (elements.statLocked) {
                elements.statLocked.textContent = folders.filter(f => f.is_locked).length;
            }
        }
    } catch (error) {
        console.error('Failed to load folders:', error);
    }
}

function renderFolders() {
    const container = elements.foldersList;
    if (!container) return;

    if (folders.length === 0) {
        container.innerHTML = `
      <div class="empty-state">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
        </svg>
        <h3>No folders yet</h3>
        <p>Click "Add Folder" to start protecting your folders</p>
      </div>
    `;
        return;
    }

    container.innerHTML = folders.map(folder => `
    <div class="folder-card ${folder.is_locked ? 'locked' : 'unlocked'}">
      <div class="folder-icon ${folder.is_locked ? 'locked' : 'unlocked'}">
        ${folder.is_locked ? `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
          </svg>
        ` : `
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 9.9-1"/>
          </svg>
        `}
      </div>
      <div class="folder-info">
        <div class="folder-name">${folder.name}</div>
        <div class="folder-path">${folder.path}</div>
      </div>
      <div class="folder-meta">
        <span>${folder.size}</span>
        <span>${folder.is_locked ? '🔒 Locked' : '🔓 Unlocked'}</span>
      </div>
      <div class="folder-actions">
        ${folder.is_locked ? `
          <button class="btn btn-primary" onclick="unlockFolder('${folder.path}')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 9.9-1"/>
            </svg>
            Unlock
          </button>
        ` : `
          <button class="btn btn-secondary" onclick="openFolder('${folder.path}')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/>
              <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open
          </button>
          <button class="btn btn-primary" onclick="lockFolder('${folder.path}')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
            Lock
          </button>
        `}
        <button class="btn btn-ghost" onclick="removeFolder('${folder.folder_id}')">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      </div>
    </div>
  `).join('');
}

async function addFolder() {
    if (!isEnrolled) {
        showToast('warning', 'Enrollment Required', 'Please enroll your face first');
        switchView('enroll');
        return;
    }

    try {
        // Open folder picker
        const folderPath = await window.electronAPI?.selectFolder();

        if (!folderPath) {
            return; // User cancelled
        }

        // Add folder to backend
        const result = await apiCall('/add-folder', {
            method: 'POST',
            body: JSON.stringify({
                path: folderPath,
                owner_id: currentUserId
            })
        });

        if (result.success) {
            showToast('success', 'Folder Added', `${result.path} added successfully`);
            loadFolders();
            loadDashboardStats();
        } else {
            showToast('error', 'Error', result.error);
        }

    } catch (error) {
        console.error('Add folder error:', error);
        showToast('error', 'Error', 'Failed to add folder');
    }
}

async function lockFolder(path) {
    try {
        const result = await apiCall('/lock-folder', {
            method: 'POST',
            body: JSON.stringify({
                path: path,
                owner_id: currentUserId
            })
        });

        if (result.success) {
            showToast('success', 'Folder Locked', result.message);
            loadFolders();
            loadDashboardStats();
        } else {
            showToast('error', 'Lock Failed', result.error);
        }

    } catch (error) {
        console.error('Lock folder error:', error);
        showToast('error', 'Error', 'Failed to lock folder');
    }
}

window.lockFolder = lockFolder;

async function unlockFolder(path) {
    // Open verification popup
    if (window.electronAPI) {
        window.electronAPI.openVerification({
            folderPath: path,
            ownerId: currentUserId
        });
    } else {
        // Fallback for development without Electron
        showToast('warning', 'Verification Required', 'Face verification popup would open here');
    }
}

window.unlockFolder = unlockFolder;

async function openFolder(path) {
    try {
        if (window.electronAPI) {
            await window.electronAPI.openFolder(path);
        }
    } catch (error) {
        console.error('Open folder error:', error);
        showToast('error', 'Error', 'Failed to open folder');
    }
}

window.openFolder = openFolder;

async function removeFolder(folderId) {
    try {
        const result = await apiCall('/remove-folder', {
            method: 'POST',
            body: JSON.stringify({
                folder_id: folderId
            })
        });

        if (result.success) {
            showToast('success', 'Folder Removed', result.message);
            loadFolders();
            loadDashboardStats();
        } else {
            showToast('error', 'Error', result.error);
        }

    } catch (error) {
        console.error('Remove folder error:', error);
        showToast('error', 'Error', 'Failed to remove folder');
    }
}

window.removeFolder = removeFolder;

// ============ Toast Notifications ============
function showToast(type, title, message, duration = 4000) {
    const container = elements.toastContainer;
    if (!container) return;

    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
    <div class="toast-icon">${icons[type] || icons.info}</div>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
  `;

    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Periodic connection check
setInterval(checkBackendConnection, 30000);
