const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('node:path');
const { spawn } = require('child_process');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

let mainWindow = null;
let verificationWindow = null;
let pythonProcess = null;

const createWindow = () => {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0a0a0f',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Load the index.html of the app
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
};

// Create verification popup window
const createVerificationWindow = (folderPath, ownerId) => {
  if (verificationWindow) {
    verificationWindow.focus();
    return;
  }

  verificationWindow = new BrowserWindow({
    width: 500,
    height: 600,
    parent: mainWindow,
    modal: true,
    frame: false,
    resizable: false,
    backgroundColor: '#0a0a0f',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  verificationWindow.loadFile(path.join(__dirname, 'verification.html'));

  // Send folder data after window loads
  verificationWindow.webContents.on('did-finish-load', () => {
    verificationWindow.webContents.send('verification-data', {
      folderPath,
      ownerId,
    });
  });

  verificationWindow.on('closed', () => {
    verificationWindow = null;
  });
};

// IPC Handlers
ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select Folder to Lock',
  });

  if (result.canceled) {
    return null;
  }

  return result.filePaths[0];
});

ipcMain.handle('open-folder', async (event, folderPath) => {
  try {
    await shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.on('open-verification', (event, data) => {
  createVerificationWindow(data.folderPath, data.ownerId);
});

ipcMain.on('close-verification', () => {
  if (verificationWindow) {
    verificationWindow.close();
  }
});

ipcMain.on('verification-success', (event, data) => {
  mainWindow.webContents.send('folder-unlocked', data);
  if (verificationWindow) {
    verificationWindow.close();
  }
});

// Window controls
ipcMain.on('window-minimize', () => {
  mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});

ipcMain.on('window-close', () => {
  mainWindow.close();
});

// Start Python backend
const startPythonBackend = () => {
  const backendPath = path.join(__dirname, '..', 'backend', 'app.py');

  // Try python3 first, then python
  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';

  pythonProcess = spawn(pythonCommand, [backendPath], {
    cwd: path.join(__dirname, '..', 'backend'),
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python Backend: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Backend Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
  });
};

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  // Optionally start Python backend automatically
  // startPythonBackend();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // Kill Python process if running
  if (pythonProcess) {
    pythonProcess.kill();
  }

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});
