// Socket.IO connection
const socket = io();

// DOM Elements
const liveVideoContainer = document.getElementById('live-video-container');
const uploadedVideoContainer = document.getElementById('uploaded-video-container');
const eventsLog = document.getElementById('events-log');
const uploadForm = document.getElementById('upload-form');
const videoInput = document.getElementById('video-input');
const startButton = document.getElementById('start-button');
const gpsData = document.getElementById('gps-data');
const criticalEventsTable = document.getElementById('critical-events-table');
const cameraSource = document.getElementById('camera-source');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const uploadLoadingBar = document.getElementById('upload-loading-bar');
const uploadProgressBar = uploadLoadingBar.querySelector('.progress-bar');

// Video feed handling
let currentVideoId = null;
let criticalEvents = [];
let stream = null;

// Blockchain Store Elements
const blockchainFileInput = document.getElementById('blockchain-file-input');
const blockchainUploadButton = document.getElementById('blockchain-upload-button');
const blockchainUploadStatus = document.getElementById('blockchain-upload-status');
const blockchainRefreshButton = document.getElementById('blockchain-refresh-button');
const blockchainFilesTable = document.getElementById('blockchain-files-table');
const blockchainListStatus = document.getElementById('blockchain-list-status');

// Handle video upload (in Critical Event Analysis tab)
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = videoInput.files[0];
    if (!file) {
        showAlert('Please select a video file to upload.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    // Show loading bar and reset progress
    uploadLoadingBar.style.display = 'block';
    uploadProgressBar.style.width = '0%';
    uploadProgressBar.setAttribute('aria-valuenow', '0');

    const xhr = new XMLHttpRequest();

    // Progress event listener
    xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            uploadProgressBar.style.width = percentComplete + '%';
            uploadProgressBar.setAttribute('aria-valuenow', percentComplete);
        }
    };

    // Load event listener (upload complete)
    xhr.onload = () => {
        uploadLoadingBar.style.display = 'none'; // Hide loading bar
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                currentVideoId = data.video_id;
                showAlert('Video uploaded successfully!', 'success');
                startButton.disabled = false;
                displayUploadedVideoFeed(currentVideoId);
            }
        } else {
            showAlert('Error uploading video. Status: ' + xhr.status, 'danger');
        }
    };

    // Error event listener
    xhr.onerror = () => {
        uploadLoadingBar.style.display = 'none'; // Hide loading bar
        showAlert('Network error during video upload.', 'danger');
    };

    // Abort event listener
    xhr.onabort = () => {
        uploadLoadingBar.style.display = 'none'; // Hide loading bar
        showAlert('Video upload aborted.', 'warning');
    };

    xhr.open('POST', '/upload');
    xhr.send(formData);
});

// Start video analysis (in Critical Event Analysis tab)
startButton.addEventListener('click', () => {
    if (!currentVideoId) {
        showAlert('Please upload a video first', 'danger');
        return;
    }

    // Start processing
    socket.emit('start_processing', { video_id: currentVideoId });
});

// Live camera handling
startCameraButton.addEventListener('click', async () => {
    const source = cameraSource.value;
    if (!source) {
        showAlert('Please select a camera source', 'warning');
        return;
    }

    try {
        // Request camera access
        const constraints = {
            video: {
                deviceId: source === 'external' ? { exact: 'external' } : undefined
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Create video element
        const video = document.createElement('video');
        video.id = 'live-video-feed';
        video.autoplay = true;
        video.playsInline = true;
        video.srcObject = stream;
        liveVideoContainer.innerHTML = '';
        liveVideoContainer.appendChild(video);

        // Start processing live feed
        socket.emit('start_live_processing', { source: source });
        
        // Update UI
        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;
        cameraSource.disabled = true;
        
        showAlert('Live camera feed started', 'success');
    } catch (error) {
        showAlert('Error accessing camera: ' + error.message, 'danger');
    }
});

stopCameraButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    // Clear video feed
    liveVideoContainer.innerHTML = '<p class="text-center text-muted">Select a camera source to begin</p>';
    
    // Stop processing
    socket.emit('stop_live_processing');
    
    // Update UI
    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;
    cameraSource.disabled = false;
    
    showAlert('Live camera feed stopped', 'info');
});

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    showAlert('Connection lost. Please refresh the page.', 'danger');
});

socket.on('new_event', (event) => {
    addEventToLog(event);
    if (event.is_critical) {
        addCriticalEvent(event);
    }
});

socket.on('gps_update', (data) => {
    updateGPSData(data);
});

socket.on('error', (data) => {
    showAlert(data.message, 'danger');
});

// Helper functions
function addEventToLog(event) {
    const eventElement = document.createElement('div');
    eventElement.className = 'event-item';
    eventElement.innerHTML = `
        <strong>${event.event_type}</strong>
        <br>
        Vehicle ID: ${event.vehicle_id}
        <br>
        Time: ${event.timestamp}
        <br>
        Status: ${event.motion_status}
        ${event.ttc !== 'N/A' ? `<br>TTC: ${event.ttc}s` : ''}
    `;
    eventsLog.insertBefore(eventElement, eventsLog.firstChild);
}

function addCriticalEvent(event) {
    criticalEvents.unshift(event);
    updateCriticalEventsTable();
}

function updateCriticalEventsTable() {
    criticalEventsTable.innerHTML = criticalEvents.map(event => `
        <tr>
            <td>${event.timestamp}</td>
            <td>${event.event_type}</td>
            <td>${event.vehicle_id}</td>
            <td>${event.motion_status}</td>
            <td>${event.ttc !== 'N/A' ? event.ttc + 's' : 'N/A'}</td>
        </tr>
    `).join('');
}

function updateGPSData(data) {
    if (data.connected) {
        gpsData.innerHTML = `
            <div class="text-success mb-2">GPS Connected</div>
            <div>Latitude: ${data.latitude.toFixed(6)}</div>
            <div>Longitude: ${data.longitude.toFixed(6)}</div>
        `;
    } else {
        gpsData.innerHTML = `
            <div class="text-danger mb-2">GPS Disconnected</div>
            <div>Waiting for GPS signal...</div>
        `;
    }
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function displayUploadedVideoFeed(videoId) {
    // Clear previous video feed
    uploadedVideoContainer.innerHTML = '';
    
    // Create video element
    const video = document.createElement('img');
    video.id = 'uploaded-video-feed';
    video.src = `/video_feed/${videoId}`;
    uploadedVideoContainer.appendChild(video);
}

// Export functions
function exportCriticalEvents() {
    if (criticalEvents.length === 0) {
        showAlert('No critical events to export', 'warning');
        return;
    }
    window.location.href = '/export_critical_events';
}

function clearExportedFiles() {
    if (confirm('Are you sure you want to clear all exported files?')) {
        fetch('/clear_exported_files', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                showAlert('All exported files cleared successfully', 'success');
            }
        })
        .catch(error => {
            showAlert('Error clearing files: ' + error.message, 'danger');
        });
    }
}

// Function to fetch and display blockchain files
async function fetchBlockchainFiles() {
    blockchainListStatus.textContent = 'Loading files...';
    try {
        const response = await fetch('/blockchain/list');
        const data = await response.json();

        blockchainFilesTable.innerHTML = ''; // Clear existing rows

        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const row = blockchainFilesTable.insertRow();
                row.innerHTML = `
                    <td>${file.id}</td>
                    <td>${file.file_name}</td>
                    <td>${new Date(file.timestamp * 1000).toLocaleString()}</td>
                    <td><button class="btn btn-sm btn-success download-btn" data-file-id="${file.id}">Download</button></td>
                `;
            });
            // Add event listeners to download buttons
            document.querySelectorAll('.download-btn').forEach(button => {
                button.addEventListener('click', (e) => {
                    const fileId = e.target.dataset.fileId;
                    window.location.href = `/blockchain/retrieve/${fileId}`;
                });
            });
            blockchainListStatus.textContent = ''; // Clear status on success
        } else {
            blockchainFilesTable.innerHTML = '<tr><td colspan="4">No files found on the blockchain.</td></tr>';
            blockchainListStatus.textContent = ''; // Clear status
        }
    } catch (error) {
        console.error('Error fetching blockchain files:', error);
        blockchainListStatus.textContent = 'Error loading files.';
        blockchainFilesTable.innerHTML = '<tr><td colspan="4">Could not load files.</td></tr>';
    }
}

// Event listener for upload button
blockchainUploadButton.addEventListener('click', async () => {
    const file = blockchainFileInput.files[0];
    if (!file) {
        showAlert('Please select a file to upload.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    blockchainUploadStatus.textContent = 'Uploading...';
    blockchainUploadButton.disabled = true;

    try {
        const response = await fetch('/blockchain/store', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            blockchainUploadStatus.textContent = data.message;
            showAlert('File uploaded successfully!', 'success');
            fetchBlockchainFiles(); // Refresh list after upload
        } else {
            blockchainUploadStatus.textContent = `Error: ${data.error}`;
            showAlert(`Error uploading file: ${data.error}`, 'danger');
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        blockchainUploadStatus.textContent = 'Error uploading file.';
        showAlert('Error uploading file.', 'danger');
    } finally {
        blockchainUploadButton.disabled = false;
        blockchainFileInput.value = ''; // Clear file input
    }
});

// Event listener for refresh button
blockchainRefreshButton.addEventListener('click', fetchBlockchainFiles);

// Load blockchain files when the blockchain tab is shown
document.querySelector('#blockchain-tab').addEventListener('shown.bs.tab', () => {
    fetchBlockchainFiles();
});

// Initial load of blockchain files when the page loads (if blockchain tab is active by default)
// You might want to adjust this based on your default active tab
// For now, it will just try to fetch on DOMContentLoaded if the tab is initially active

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    startButton.disabled = true;
    stopCameraButton.disabled = true;
    
    // Initialize Bootstrap tabs
    const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabElList.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });
}); 