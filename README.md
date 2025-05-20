

---

````markdown
# Eagle Eye – Vehicle Critical Events Analysis System

Eagle Eye is a real-time intelligent traffic analysis system that detects and classifies critical driving events using both **rule-based logic** and **machine learning models**. It supports both **uploaded video analysis** and **live camera analysis**, with blockchain-based storage for audit logs.

---

## 🔧 Setup and Running Guide

### 📦 Prerequisites

- Python 3.8+
- Node.js and npm
- CUDA-compatible GPU (optional but recommended)
- Webcam for live streaming analysis

---

### ⚙️ Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
````

---

### 🤖 Step 2: Model Setup

Ensure the following models are available:

* `yolov8n.pt` – YOLOv8 model (will auto-download if missing)
* `app/models/scaler.pkl`
* `app/models/frontier_classifier.pkl`
* `app/frontier_anomaly_model.pkl`

---

### 🗄️ Step 3: Database Setup

* Uses SQLite with SQLAlchemy.
* Database is auto-created on first run.

---

### ▶️ Step 4: Running the Application

```bash
python run.py
```

Then open your browser at:
**[http://localhost:5000](http://localhost:5000)**

---

## 📽️ How to Use

### 🔍 1. Video Analysis

* Upload `.mp4` files in the Critical Event Analysis tab.
* Detected events are processed and displayed in a table and event log.

### 🎥 2. Live Stream Analysis

* Select a webcam from the dropdown.
* Critical events will be detected in real time while the camera is active.

### 📊 3. Critical Events Table

* Red: Collisions, Harsh Braking
* Yellow: Detected anomalies
* Light blue: Frontier vehicles
* Gray: Tracked non-critical vehicles

### 📤 4. Exporting Results

* Export critical events to Excel
* Clear exported files using built-in buttons

---

## 🔗 Blockchain Integration (via Ganache)

### 📥 1. Install Ganache

Download: [https://trufflesuite.com/ganache/](https://trufflesuite.com/ganache/)

### ⚙️ 2. Ganache Configuration

* Port: `7545`
* Network ID: `1337`
* Automine: On
* Gas Limit: `6721975`

### 🌐 3. Configure Environment

Create a `.env` file:

```bash
MNEMONIC="your ganache mnemonic"
INFURA_PROJECT_ID="your infura id" # optional
```

Install HD Wallet:

```bash
npm install @truffle/hdwallet-provider dotenv
```

### 📦 4. Compile & Deploy Contracts

```bash
npx truffle compile
npx truffle migrate --network development
```

### 🧠 5. Smart Contract: `DataStorage.sol`

* `storeZipFile(filename, zipData)`
* `getFileNames()`
* `getZipFile(fileId)`

---

## 🛠️ Troubleshooting

| Issue               | Solution                                             |
| ------------------- | ---------------------------------------------------- |
| `500 Upload Error`  | Ensure enough disk space & video file is valid       |
| Camera not detected | Grant camera permissions, ensure device is connected |
| ML model errors     | Check if `.pkl` files exist with correct paths       |
| Contract error      | Ensure Ganache is running and migrated               |

---

## 💡 Additional Notes

* WebSocket used for real-time updates
* ML Models: Isolation Forest, RandomForest, YOLO, DeepSORT, Kalman Filter
* Blockchain contract stores critical event ZIPs securely

---

## 🙋 Contact

For questions or contributions, open an issue or contact the maintainer via GitHub.

---

```

```
