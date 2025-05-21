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

// Add these functions after the existing DOM Elements
const storedVideosList = null;
const clearVideosButton = null;

// Pedestrian Analysis Variables
let pedestrianEvents = [];
const pedestrianVideoInput = document.getElementById('pedestrian-video-input');
const pedestrianUploadForm = document.getElementById('pedestrian-upload-form');
const pedestrianVideoContainer = document.getElementById('pedestrian-video-container');
const pedestrianEventsTable = document.getElementById('pedestrian-events-table');
const pedestrianUploadLoadingBar = document.getElementById('pedestrian-upload-loading-bar');
const pedestrianProgressBar = pedestrianUploadLoadingBar.querySelector('.progress-bar');

async function populateCameraOptions() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true }); // ask permission

        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSource.innerHTML = '<option value="">Select a camera...</option>';

        let index = 1;
        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${index++}`;
                cameraSource.appendChild(option);
            }
        });
    } catch (err) {
        console.error('Camera permission error:', err);
        showAlert('Camera access required to list devices.', 'danger');
    }
}

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

    // Validate file type
    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file.', 'warning');
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
        uploadLoadingBar.style.display = 'none';
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                currentVideoId = data.video_id;
                showAlert('Video uploaded successfully! Starting analysis...', 'success');
                displayUploadedVideoFeed();
                
                // Automatically start processing after successful upload
                socket.emit('start_processing', { video_id: currentVideoId });
            }
        } else {
            let errorMessage = 'Error uploading video. ';
            try {
                const data = JSON.parse(xhr.responseText);
                errorMessage += data.error || `Status: ${xhr.status}`;
            } catch (e) {
                errorMessage += `Status: ${xhr.status}`;
            }
            showAlert(errorMessage, 'danger');
        }
    };

    // Error event listener
    xhr.onerror = () => {
        uploadLoadingBar.style.display = 'none';
        showAlert('Network error during video upload. Please check your connection and try again.', 'danger');
    };

    // Abort event listener
    xhr.onabort = () => {
        uploadLoadingBar.style.display = 'none';
        showAlert('Video upload was cancelled.', 'warning');
    };

    xhr.open('POST', '/upload');
    xhr.send(formData);
});

// Start video analysis (in Critical Event Analysis tab)
// startButton.addEventListener('click', () => {
//     if (!currentVideoId) {
//         showAlert('Please upload a video first', 'danger');
//         return;
//     }

//     // Start processing
//     socket.emit('start_processing', { video_id: currentVideoId });
// });

startCameraButton.addEventListener('click', async () => {
    const selectedDeviceId = cameraSource.value;
    if (!selectedDeviceId) {
        showAlert('Please select a camera device', 'warning');
        return;
    }

    try {
        const constraints = {
            video: {
                deviceId: { exact: selectedDeviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);

        const video = document.createElement('video');
        video.id = 'live-video-feed';
        video.autoplay = true;
        video.playsInline = true;
        video.srcObject = stream;

        liveVideoContainer.innerHTML = '';
        liveVideoContainer.appendChild(video);

        socket.emit('start_live_processing', { deviceId: selectedDeviceId });

        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;
        cameraSource.disabled = true;

        showAlert('Live camera feed started', 'success');
    } catch (error) {
        showAlert('Error accessing camera: ' + error.message, 'danger');
    }
});


// Live camera handling
// startCameraButton.addEventListener('click', async () => {
//     const source = cameraSource.value;
//     if (!source) {
//         showAlert('Please select a camera source', 'warning');
//         return;
//     }

//     try {
//         // Request camera access
//         const constraints = {
//             video: {
//                 deviceId: source === 'external' ? { exact: 'external' } : undefined
//             }
//         };
        
//         stream = await navigator.mediaDevices.getUserMedia(constraints);
        
//         // Create video element
//         const video = document.createElement('video');
//         video.id = 'live-video-feed';
//         video.autoplay = true;
//         video.playsInline = true;
//         video.srcObject = stream;
//         liveVideoContainer.innerHTML = '';
//         liveVideoContainer.appendChild(video);

//         // Start processing live feed
//         socket.emit('start_live_processing', { source: source });
        
//         // Update UI
//         startCameraButton.disabled = true;
//         stopCameraButton.disabled = false;
//         cameraSource.disabled = true;
        
//         showAlert('Live camera feed started', 'success');
//     } catch (error) {
//         showAlert('Error accessing camera: ' + error.message, 'danger');
//     }
// });

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
    // Add all events to the appropriate real-time analysis table based on active tab
    const criticalTab = document.getElementById('critical-tab');
    const liveTab = document.getElementById('livestream-tab');
    
    let targetTableBody = null;
    if (criticalTab && criticalTab.classList.contains('active') && criticalEventsTable) {
        targetTableBody = criticalEventsTable.querySelector('tbody');
    } else if (liveTab && liveTab.classList.contains('active') && liveEventsTableBody) {
        targetTableBody = liveEventsTableBody;
    }

    if (!targetTableBody) {
        // If no active tab with a corresponding table is found, do nothing
        return;
    }

    const row = document.createElement('tr');
    
    // Determine row class based on event type and motion status
    let rowClass = '';
    if (event.motion_status === 'Collided' || event.motion_status === 'Harsh Braking' || event.motion_status === 'Sudden Stop Detected!' || event.event_type === 'Near Collision') {
        rowClass = 'table-danger'; // Red for critical motion statuses or near collision
    } else if (event.event_type && event.event_type.includes('Anomaly')) {
        rowClass = 'table-warning'; // Yellow for anomalies
    } else if (event.event_type === 'Frontier') {
        rowClass = 'table-info'; // Light blue for non-critical frontier vehicles
    } else { // Default case for 'Tracked' and other events
         rowClass = 'table-active'; // Gray for regular tracked vehicles
    }
    row.className = rowClass;

    row.innerHTML = `
        <td>${event.id !== undefined ? event.id : 'N/A'}</td>
        <td>${event.timestamp !== undefined ? event.timestamp : 'N/A'}</td>
        <td>${event.event_type !== undefined ? event.event_type : 'N/A'}</td>
        <td>${event.vehicle_id !== undefined ? event.vehicle_id : 'N/A'}</td>
        <td>${event.motion_status !== undefined ? event.motion_status : 'N/A'}</td>
        <td>${event.ttc !== undefined && event.ttc !== null && event.ttc !== 'N/A' ? parseFloat(event.ttc).toFixed(2) : 'N/A'}</td>
        <td>${event.latitude !== undefined && event.longitude !== undefined ? `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}` : 'N/A'}</td>
    `;
    targetTableBody.insertBefore(row, targetTableBody.firstChild); // Add to the top of the table
});

// socket.on('gps_update', (data) => {
//     console.log("📡 GPS Data Received:", data);  // <== ADD THIS LINE
//     // Only update GPS data display if the Live Streaming tab is active
//     const liveTab = document.getElementById('livestream-tab');
//     if (liveTab && liveTab.classList.contains('active')) {
//         updateGPSData(data);
//     }
// });

socket.on('gps_update', (data) => {
    console.log("📡 GPS Data Received:", data);
    updateGPSData(data);  // Always update regardless of tab
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
            <td>${event.latitude.toFixed(6)}, ${event.longitude.toFixed(6)}</td>  <!-- 👈 THIS LINE -->
        </tr>
    `).join('');
}

function updateGPSData(data) {
    console.log('updateGPSData called with data:', data);
    if (!gpsData) {
        console.error('GPS Data element not found!');
        return;
    }
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

// function displayUploadedVideoFeed() {
//     // Clear previous video feed
//     uploadedVideoContainer.innerHTML = '';
    
//     // Create video element
//     const video = document.createElement('img');
//     video.id = 'uploaded-video-feed';
//     video.src = '/video_feed';  // Updated to use the new endpoint
//     uploadedVideoContainer.appendChild(video);
// }


function displayUploadedVideoFeed() {
    uploadedVideoContainer.innerHTML = '';

    // Add timestamp to force refresh
    const timestamp = new Date().getTime();
    const video = document.createElement('img');
    video.id = 'uploaded-video-feed';
    video.src = `/video_feed?t=${timestamp}`; // 💡 this avoids caching
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
    // Remove this line as the start button is removed
    // startButton.disabled = true;
    stopCameraButton.disabled = true;

    // Initialize Bootstrap tabs
    const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabElList.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });

    populateCameraOptions(); // <== call this to populate the camera dropdown

    // Get the critical events table body element once the DOM is loaded
    const criticalEventsTableBody = document.getElementById('critical-events-table');
    // Add a check in case the element is not found for some reason
    if (!criticalEventsTableBody) {
        console.error("Error: Table body with ID 'critical-events-table' not found.");
        // Do not attach the new_event listener if the table body is not found
        // return; // Keep listening for events even if critical analysis table is not present
    }

    // Get the live events table body element once the DOM is loaded
    const liveEventsTableBody = document.getElementById('live-events-table-body');
     if (!liveEventsTableBody) {
        console.error("Error: Table body with ID 'live-events-table-body' not found.");
        // Keep listening for events even if live analysis table is not present
    }

    // Move Socket.IO event handlers inside DOMContentLoaded
    // Socket.IO event handlers
    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        showAlert('Connection lost. Please refresh the page.', 'danger');
    });

    socket.on('new_event', (event) => {
        // Add all events to the appropriate real-time analysis table based on active tab
        const criticalTab = document.getElementById('critical-tab');
        const liveTab = document.getElementById('livestream-tab');
        
        let targetTableBody = null;
        if (criticalTab && criticalTab.classList.contains('active') && criticalEventsTableBody) {
            targetTableBody = criticalEventsTableBody;
        } else if (liveTab && liveTab.classList.contains('active') && liveEventsTableBody) {
            targetTableBody = liveEventsTableBody;
        }

        if (!targetTableBody) {
            // If no active tab with a corresponding table is found, do nothing
            return;
        }

        const row = document.createElement('tr');
        
        // Determine row class based on event type and motion status
        let rowClass = '';
        if (event.motion_status === 'Collided' || event.motion_status === 'Harsh Braking' || event.motion_status === 'Sudden Stop Detected!' || event.event_type === 'Near Collision') {
            rowClass = 'table-danger'; // Red for critical motion statuses or near collision
        } else if (event.event_type && event.event_type.includes('Anomaly')) {
            rowClass = 'table-warning'; // Yellow for anomalies
        } else if (event.event_type === 'Frontier') {
            rowClass = 'table-info'; // Light blue for non-critical frontier vehicles
        } else { // Default case for 'Tracked' and other events
             rowClass = 'table-active'; // Gray for regular tracked vehicles
        }
        row.className = rowClass;

        row.innerHTML = `
            <td>${event.id !== undefined ? event.id : 'N/A'}</td>
            <td>${event.timestamp !== undefined ? event.timestamp : 'N/A'}</td>
            <td>${event.event_type !== undefined ? event.event_type : 'N/A'}</td>
            <td>${event.vehicle_id !== undefined ? event.vehicle_id : 'N/A'}</td>
            <td>${event.motion_status !== undefined ? event.motion_status : 'N/A'}</td>
            <td>${event.ttc !== undefined && event.ttc !== null && event.ttc !== 'N/A' ? parseFloat(event.ttc).toFixed(2) : 'N/A'}</td>
            <td>${event.latitude !== undefined && event.longitude !== undefined ? `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}` : 'N/A'}</td>
        `;
        targetTableBody.insertBefore(row, targetTableBody.firstChild); // Add to the top of the table
        
        // Show alert for critical events based on the corrected frontend logic
        // Removed this block to stop pop-up notifications
        // if (event.motion_status === 'Collided' || event.motion_status === 'Harsh Braking' || event.motion_status === 'Sudden Stop Detected!' || event.event_type === 'Near Collision') {
        //      showAlert(`Critical Event: ${event.event_type} - ${event.motion_status}`, 'danger');
        // }
    });

    socket.on('gps_update', (data) => {
        console.log("📡 GPS Data Received:", data);  // <== ADD THIS LINE
        // Only update GPS data display if the Live Streaming tab is active
        const liveTab = document.getElementById('livestream-tab');
        if (liveTab && liveTab.classList.contains('active')) {
            updateGPSData(data);
        }
    });


    socket.on('error', (data) => {
        showAlert(data.message, 'danger');
    });

});

// Handle live stream frames
socket.on('frame', (data) => {
    const videoFeed = document.getElementById('live-video-feed');
    if (videoFeed) {
        const blob = new Blob([data.frame], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        videoFeed.src = url;
    }
});

// Pedestrian Analysis Functions
function addPedestrianEvent(event) {
    pedestrianEvents.unshift(event);
    updatePedestrianEventsTable();
}

function updatePedestrianEventsTable() {
    if (!pedestrianEventsTable) return;
    
    pedestrianEventsTable.innerHTML = pedestrianEvents.map(event => `
        <tr class="${getPedestrianEventRowClass(event)}">
            <td>${event.id}</td>
            <td>${event.timestamp}</td>
            <td>${event.pedestrian_id}</td>
            <td>${event.intent_score.toFixed(2)}</td>
            <td>${event.speed.toFixed(2)} px/frame</td>
            <td>${getPedestrianStatus(event.intent_score)}</td>
            <td>(${event.location.x}, ${event.location.y})</td>
        </tr>
    `).join('');
}

function getPedestrianEventRowClass(event) {
    if (event.intent_score >= 0.7) {
        return 'table-danger'; // Red for high intent
    } else if (event.intent_score >= 0.5) {
        return 'table-warning'; // Orange for medium intent
    }
    return 'table-success'; // Green for low intent
}

function getPedestrianStatus(intentScore) {
    if (intentScore >= 0.7) {
        return 'High Risk';
    } else if (intentScore >= 0.5) {
        return 'Medium Risk';
    }
    return 'Low Risk';
}

function exportPedestrianEvents() {
    if (pedestrianEvents.length === 0) {
        showAlert('No pedestrian events to export', 'warning');
        return;
    }
    window.location.href = '/export_pedestrian_events';
}

function clearPedestrianFiles() {
    fetch('/clear_pedestrian_files', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Pedestrian files cleared successfully', 'success');
        } else {
            showAlert('Failed to clear pedestrian files', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error clearing pedestrian files', 'danger');
    });
}

// Handle pedestrian video upload
pedestrianUploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const file = pedestrianVideoInput.files[0];
    if (!file) {
        showAlert('Please select a video file', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    pedestrianUploadLoadingBar.style.display = 'block';
    pedestrianProgressBar.style.width = '0%';

    fetch('/upload_pedestrian_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Video uploaded successfully', 'success');
            pedestrianVideoContainer.innerHTML = `
                <video id="pedestrian-video" controls class="w-100">
                    <source src="${data.video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
        } else {
            showAlert(data.error || 'Failed to upload video', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error uploading video', 'danger');
    })
    .finally(() => {
        pedestrianUploadLoadingBar.style.display = 'none';
    });
});

// Socket event handler for pedestrian events
socket.on('pedestrian_event', (event) => {
    addPedestrianEvent(event);
});

socket.on("new_event", function(data) {
    const tbody = document.getElementById("live-events-table-body");
    if (!tbody) return;

    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${data.id ?? 'N/A'}</td>
        <td>${data.timestamp}</td>
        <td>${data.event_type}</td>
        <td>${data.vehicle_id}</td>
        <td>${data.motion_status}</td>
        <td>${data.ttc}</td>
        <td>${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}</td>
    `;

    tbody.prepend(row);  // Add to top of the table
});

