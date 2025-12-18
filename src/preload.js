const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
    // Folder operations
    selectFolder: () => ipcRenderer.invoke('select-folder'),
    openFolder: (path) => ipcRenderer.invoke('open-folder', path),

    // Verification window
    openVerification: (data) => ipcRenderer.send('open-verification', data),
    closeVerification: () => ipcRenderer.send('close-verification'),
    onVerificationData: (callback) => {
        ipcRenderer.on('verification-data', (event, data) => callback(data));
    },
    verificationSuccess: (data) => ipcRenderer.send('verification-success', data),
    onFolderUnlocked: (callback) => {
        ipcRenderer.on('folder-unlocked', (event, data) => callback(data));
    },

    // Window controls
    minimize: () => ipcRenderer.send('window-minimize'),
    maximize: () => ipcRenderer.send('window-maximize'),
    close: () => ipcRenderer.send('window-close'),
});

// Expose platform info
contextBridge.exposeInMainWorld('platform', {
    isWindows: process.platform === 'win32',
    isMac: process.platform === 'darwin',
    isLinux: process.platform === 'linux',
});
