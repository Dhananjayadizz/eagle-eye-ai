from flask import Flask, request, render_template, jsonify, Response, send_file
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from app.core.sort import Sort
from app.core.gps_module import get_gps_data
from app.core.vehicle_tracker import VehicleTracker
from app.core.motion_detection import detect_motion_changes
import joblib
import torch
import io
import atexit
import logging
import pickle
import openpyxl
from pathlib import Path
import math
import uuid
import threading
import serial
import time
import random
import shutil
import re
# Import blockchain blueprint
from app.blockchain import blockchain_bp


from threading import Lock

# Global GPS data and lock
gps_data = {
    "latitude": 0.0,
    "longitude": 0.0,
    "connected": False
}
gps_lock = Lock()


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, logger=True, engineio_logger=True)

# Register the blockchain blueprint
app.register_blueprint(blockchain_bp, url_prefix='/blockchain')

UPLOAD_FOLDER = "uploads"
EXPORT_DIR = "exports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

PIXELS_PER_METER = 0.1
vehicle_history = {}
collided_vehicles = set()
collision_cooldown = {}
ego_gps_history = {}  # To store GPS history for the ego vehicle
frontier_gps_history = {}  # To store GPS history for the frontier vehicle

# Global variables for GPS data and serial communication
# current_latitude = 0.0
# current_longitude = 0.0
# gps_connected = False
serial_port = None
SERIAL_PORT_NAME = 'COM5' # Set the serial port name to COM5
BAUDRATE = 115200 # Match the Serial.begin() baud rate in your Arduino sketch

# Replace the MAX_STORED_VIDEOS constant with a single video file name
CURRENT_VIDEO_FILE = "current_video.mp4"

# def read_gps_data_from_serial(port, baudrate):
#     global current_latitude, current_longitude, gps_connected, serial_port
#     logger.info(f"Attempting to connect to serial port {port} at {baudrate} baud.")

#     try:
#         serial_port = serial.Serial(port, baudrate)
#         gps_connected = True
#         logger.info(f"Successfully connected to serial port {port}.")
#         line = ''
#         lat = None
#         lon = None

#         while gps_connected:
#             if serial_port.in_waiting > 0:
#                 char = serial_port.read().decode('utf-8', errors='ignore')
#                 line += char

#                 if char == '\n':
#                     clean_line = line.strip()
#                     logger.debug(f"Received serial data: {clean_line}")

#                     if clean_line.startswith("Latitude:"):
#                         try:
#                             lat = float(clean_line.split(":")[1].strip())
#                         except ValueError:
#                             logger.error(f"Could not parse latitude from: {clean_line}")

#                     elif clean_line.startswith("Longitude:"):
#                         try:
#                             lon = float(clean_line.split(":")[1].strip())
#                         except ValueError:
#                             logger.error(f"Could not parse longitude from: {clean_line}")

#                     elif "Waiting for GPS signal..." in clean_line:
#                         logger.info("GPS signal not acquired yet.")
#                         gps_connected = False
#                         lat = lon = None
#                         socketio.emit('gps_update', {
#                             'latitude': 0.0,
#                             'longitude': 0.0,
#                             'connected': False
#                         })

#                     # Emit only when both lat and lon are ready
#                     if lat is not None and lon is not None:
#                         current_latitude = lat
#                         current_longitude = lon
#                         socketio.emit('gps_update', {
#                             'latitude': lat,
#                             'longitude': lon,
#                             'connected': True
#                         })
#                         print(f"📡 Emitting GPS: lat={current_latitude}, lon={current_longitude}")
#                         lat = lon = None  # Reset for next round

#                     line = ''

#             time.sleep(0.01)

#     except serial.SerialException as e:
#         logger.error(f"Serial port error: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error in GPS thread: {e}")

#     gps_connected = False
#     current_latitude = 0.0
#     current_longitude = 0.0
#     socketio.emit('gps_update', {
#         'latitude': 0.0,
#         'longitude': 0.0,
#         'connected': False
#     })


# def read_gps_data_from_serial(port, baudrate):
#     global serial_port
#     logger.info(f"Connecting to GPS on {port} at {baudrate}...")

#     try:
#         serial_port = serial.Serial(port, baudrate, timeout=1)
#         logger.info("Serial port opened.")
#         with gps_lock:
#             gps_data["connected"] = True

#         line = ''
#         lat, lon = None, None

#         while True:
#             if serial_port.in_waiting > 0:
#                 char = serial_port.read().decode('utf-8', errors='ignore')
#                 line += char

#                 if char == '\n':
#                     clean_line = line.strip()
#                     logger.debug(f"📥 Raw line: {clean_line}")

#                     if clean_line.startswith("Latitude:"):
#                         try:
#                             lat = float(clean_line.split(":")[1].strip())
#                             logger.debug(f"✅ Parsed Latitude: {lat}")
#                         except Exception as e:
#                             logger.error(f"❌ Failed to parse latitude: {clean_line} | {e}")

#                     elif clean_line.startswith("Longitude:"):
#                         try:
#                             lon = float(clean_line.split(":")[1].strip())
#                             logger.debug(f"✅ Parsed Longitude: {lon}")
#                         except Exception as e:
#                             logger.error(f"❌ Failed to parse longitude: {clean_line} | {e}")

#                     elif "Waiting for GPS signal..." in clean_line:
#                         logger.warning("⚠️ GPS signal not locked yet.")
#                         with gps_lock:
#                             gps_data.update({
#                                 "latitude": 0.0,
#                                 "longitude": 0.0,
#                                 "connected": False
#                             })
#                         socketio.emit('gps_update', gps_data)

#                     # Emit only if both values were found
#                     if lat is not None and lon is not None:
#                         with gps_lock:
#                             gps_data.update({
#                                 "latitude": lat,
#                                 "longitude": lon,
#                                 "connected": True
#                             })
#                         logger.info(f"📡 Emitting GPS: lat={lat}, lon={lon}")
#                         socketio.emit('gps_update', gps_data)
#                         print(f"📡 [Python] GPS Emitted: {gps_data['latitude']}, {gps_data['longitude']}")
#                         lat, lon = None, None  # reset

#                     line = ''  # clear after processing

#             time.sleep(0.01)

#     except serial.SerialException as e:
#         logger.error(f"❌ Serial port error: {e}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected GPS thread error: {e}")

#     with gps_lock:
#         gps_data.update({
#             "latitude": 0.0,
#             "longitude": 0.0,
#             "connected": False
#         })
#     socketio.emit('gps_update', gps_data)




def read_gps_data_from_serial(port, baudrate):
    global serial_port
    logger.info(f"Connecting to GPS on {port} at {baudrate}...")

    try:
        serial_port = serial.Serial(port, baudrate, timeout=1)
        logger.info("Serial port opened.")
        with gps_lock:
            gps_data["connected"] = True

        while True:
            if serial_port.in_waiting > 0:
                line = serial_port.readline().decode('utf-8', errors='ignore').strip()
                logger.debug(f"Raw GPS line: {line}")

                lat_match = re.search(r'Latitude:(-?\d+\.\d+)', line)
                lon_match = re.search(r'Longitude:(-?\d+\.\d+)', line)

                if lat_match:
                    with gps_lock:
                        gps_data['latitude'] = float(lat_match.group(1))
                        gps_data['connected'] = True

                if lon_match:
                    with gps_lock:
                        gps_data['longitude'] = float(lon_match.group(1))
                        gps_data['connected'] = True

                    # Emit once both are likely to be updated
                    socketio.emit('gps_update', gps_data)

            time.sleep(0.1)

    except Exception as e:
        logger.error(f"GPS Serial Error: {e}")
        with gps_lock:
            gps_data.update({
                "latitude": 0.0,
                "longitude": 0.0,
                "connected": False
            })
        socketio.emit('gps_update', gps_data)






# def read_gps_data_from_serial(port, baudrate):
#     global current_latitude, current_longitude, gps_connected, serial_port
#     logger.info(f"Attempting to connect to serial port {port} at {baudrate} baud.")
#     try:
#         serial_port = serial.Serial(port, baudrate)
#         gps_connected = True
#         logger.info(f"Successfully connected to serial port {port}.")
#         line = ''
#         while gps_connected:
#             if serial_port.in_waiting > 0:
#                 char = serial_port.read().decode('utf-8', errors='ignore')
#                 line += char
#                 if char == '\n':
#                     logger.debug(f"Received serial data: {line.strip()}")
#                     if line.startswith("Latitude:"):
#                         try:
#                             current_latitude = float(line.split(":")[1].strip())
#                             # Emit GPS update when new latitude is received
#                             socketio.emit('gps_update', {
#                                 'latitude': current_latitude,
#                                 'longitude': current_longitude,
#                                 'connected': True
#                             })
#                         except ValueError:
#                             logger.error(f"Could not parse latitude from line: {line.strip()}")
#                     elif line.startswith("Longitude:"):
#                         try:
#                             current_longitude = float(line.split(":")[1].strip())
#                             # Emit GPS update when new longitude is received
#                             socketio.emit('gps_update', {
#                                 'latitude': current_latitude,
#                                 'longitude': current_longitude,
#                                 'connected': True
#                             })
#                         except ValueError:
#                             logger.error(f"Could not parse longitude from line: {line.strip()}")
#                     elif "Waiting for GPS signal..." in line:
#                         logger.info("GPS signal not acquired yet.")
#                         socketio.emit('gps_update', {
#                             'latitude': 0.0,
#                             'longitude': 0.0,
#                             'connected': False
#                         })
#                     line = ''
#             time.sleep(0.01) # Small delay to prevent high CPU usage
#     except serial.SerialException as e:
#         logger.error(f"Serial port error: {e}")
#         gps_connected = False
#         current_latitude = 0.0
#         current_longitude = 0.0
#         socketio.emit('gps_update', {
#             'latitude': 0.0,
#             'longitude': 0.0,
#             'connected': False
#         })
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during serial reading: {e}")
#         gps_connected = False
#         current_latitude = 0.0
#         current_longitude = 0.0
#         socketio.emit('gps_update', {
#             'latitude': 0.0,
#             'longitude': 0.0,
#             'connected': False
#         })

class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, nullable=False)
    event_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    x1 = db.Column(db.Integer)
    y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer)
    y2 = db.Column(db.Integer)
    ttc = db.Column(db.Float, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    motion_status = db.Column(db.String(50), nullable=True)

# Clear the exports directory on startup
def clear_exports_directory():
    try:
        for filename in os.listdir(EXPORT_DIR):
            file_path = os.path.join(EXPORT_DIR, filename)
            if os.path.isfile(file_path) and filename.endswith('.xlsx'):
                os.unlink(file_path)  # Use os.unlink to remove the file
        logger.info(f"Cleared contents of {EXPORT_DIR} on startup.")
    except Exception as e:
        logger.error(f"Error clearing exports directory on startup: {e}")

# Within the application context, clear exports and create database tables
with app.app_context():
    clear_exports_directory()
    db.create_all()

# Load models
model = YOLO("yolov8n.pt").to(device)
tracker = Sort()
kalman_tracker = VehicleTracker()
anomaly_model = joblib.load("app/frontier_anomaly_model.pkl")
scaler = joblib.load("app/models/scaler.pkl")
try:
    with open("app/models/frontier_classifier.pkl", "rb") as f:
        frontier_clf = pickle.load(f)
    logger.info("Frontier vehicle classification model loaded successfully.")
except FileNotFoundError:
    logger.error("Frontier classification model 'frontier_classifier.pkl' not found.")
    raise
except Exception as e:
    logger.error(f"Error loading frontier classification model: {e}")
    raise

# Haversine formula to calculate distance between two GPS coordinates (in meters)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance

# Calculate speed from GPS coordinates over time using frame count
def calculate_speed_from_gps(gps_history, lat, lon, frame_count, frame_time):
    key = "ego"  # Use a fixed key for the ego vehicle
    if key not in gps_history:
        gps_history[key] = {"last_lat": lat, "last_lon": lon, "last_frame": frame_count, "speed": 40.0}
        return 40.0

    last_lat = gps_history[key]["last_lat"]
    last_lon = gps_history[key]["last_lon"]
    last_frame = gps_history[key]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time

    if time_diff <= 0:
        logger.info(f"Time difference zero or negative for ego vehicle")
        return gps_history[key]["speed"]

    distance = haversine_distance(last_lat, last_lon, lat, lon)
    speed_mps = distance / time_diff
    speed_kmh = speed_mps * 3.6
    speed_kmh = max(0, min(120, speed_kmh))

    alpha = 0.7
    smoothed_speed = alpha * speed_kmh + (1 - alpha) * gps_history[key]["speed"]
    gps_history[key]["speed"] = smoothed_speed
    gps_history[key]["last_lat"] = lat
    gps_history[key]["last_lon"] = lon
    gps_history[key]["last_frame"] = frame_count
    logger.info(f"Calculated speed for ego vehicle: {smoothed_speed} km/h")
    return smoothed_speed


# Draw a cyan speedometer with a transparent background as a 270-degree arc with numbers on the arc
def draw_speedometer(frame, speed, center_x=None, center_y=None, radius=60):
    CYAN = (95, 189, 255)  # Define cyan color explicitly

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Set the speedometer position to the bottom-right corner
    margin = 20  # Margin from the edges
    center_x = width - radius - margin  # Position center_x near the right edge
    center_y = height - radius - margin  # Position center_y near the bottom edge

    # Draw the outer arc of the speedometer (270 degrees, from 315° to 225° counterclockwise)
    start_angle = 315    # Start at 7:30 position (315°)
    end_angle = 225      # End at 4:30 position (225°), covering 270° counterclockwise
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, CYAN, 2)

    # Draw speed markers and numbers on the arc
    for speed_mark in range(0, 121, 20):
        # Map speed (0-120) to angle (315° to 225° counterclockwise), starting from 0 at 315° to 120 at 225°
        angle = math.radians(315 - (speed_mark / 120.0) * 270)  # From 315° to 225° (270° range)
        x1 = int(center_x + (radius - 5) * math.cos(angle))  # Inner point of the marker
        y1 = int(center_y - (radius - 5) * math.sin(angle))
        x2 = int(center_x + radius * math.cos(angle))  # Outer point of the marker (on the arc)
        y2 = int(center_y - radius * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), CYAN, 1)

        # Place the number exactly on the arc
        label_x = int(center_x + radius * math.cos(angle))  # Position exactly on the arc
        label_y = int(center_y - radius * math.sin(angle))

        # Adjust text position based on angle to center the numbers
        text = str(speed_mark)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # Calculate offset to center the text on the arc
        offset_x = -text_width // 2  # Center the text horizontally
        offset_y = text_height // 2  # Center the text vertically
        adjusted_x = label_x + offset_x
        adjusted_y = label_y + offset_y

        # Add a subtle black outline to the text for better visibility
        cv2.putText(frame, text, (adjusted_x, adjusted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, text, (adjusted_x, adjusted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)  # Cyan text

    # Draw the needle in cyan
    speed = min(max(speed, 0), 120)  # Clamp speed between 0 and 120
    angle = math.radians(315 - (speed / 120.0) * 270)  # Map speed from 315° (0 km/h) to 225° (120 km/h)
    needle_length = radius - 10
    needle_x = int(center_x + needle_length * math.cos(angle))
    needle_y = int(center_y - needle_length * math.sin(angle))
    cv2.line(frame, (center_x, center_y), (needle_x, needle_y), CYAN, 2)

    # Draw the speed text in the top-right corner of the video feed
    speed_text = f"{int(speed)} km/h"
    (text_width, text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_pos = (width - text_width - 20, 30)  # Position in top-right corner (20 pixels from right, 30 from top)
    cv2.putText(frame, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
    cv2.putText(frame, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)  # Cyan text


def calculate_ttc(ego_speed, frontier_speed, distance):
    if frontier_speed <= ego_speed or distance <= 0:
        return float('inf')
    relative_speed = (ego_speed - frontier_speed) / 3.6
    return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')

def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
    if track_id not in vehicle_history:
        vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
        return 40.0
    last_y = vehicle_history[track_id]["last_y"]
    last_frame = vehicle_history[track_id]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time
    if time_diff > 0:
        displacement = last_y - y_center
        speed_pixels_per_sec = displacement / time_diff
        speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
        speed_kmh = speed_mps * 3.6
        alpha = 0.7
        new_speed = max(0, min(120, speed_kmh))
        smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
        vehicle_history[track_id]["speed"] = smoothed_speed
    vehicle_history[track_id]["last_y"] = y_center
    vehicle_history[track_id]["last_frame"] = frame_count
    return vehicle_history[track_id]["speed"]


# def detect_lanes(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 100, 200)
#     height, width = frame.shape[:2]
#     mask = np.zeros_like(edges)
#     roi_vertices = np.array([
#         [0, height * 0.6], [width, height * 0.6], [width, height], [0, height]
#     ], np.int32)
#     cv2.fillPoly(mask, [roi_vertices], 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([180, 30, 255])
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)
#     masked_edges = cv2.bitwise_and(masked_edges, masked_edges, mask=white_mask)
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=30)
#     lane_lines = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             lane_lines.append((x1, y1, x2, y2))
#     return lane_lines

def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]

    # Color masks for white and yellow lanes
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    lower_yellow = np.array([15, 50, 100])
    upper_yellow = np.array([35, 255, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150)

    # Apply region of interest (trapezoid mask)
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[  # Trapezoid mask
        [width * 0.1, height * 0.9],
        [width * 0.4, height * 0.55],
        [width * 0.6, height * 0.55],
        [width * 0.9, height * 0.9]
    ]], np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    masked_edges = cv2.bitwise_and(masked_edges, color_mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=40)

    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:  # filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))
    return lane_lines


# def get_ego_lane_bounds(lane_lines, width, height):
#     if not lane_lines:
#         return 0, width
#     left_lane_x = width
#     right_lane_x = 0
#     left_lines = []
#     right_lines = []
#     for x1, y1, x2, y2 in lane_lines:
#         if y1 != y2:
#             slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
#             x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
#             if slope > 0.5:
#                 right_lines.append(x_bottom)
#             elif slope < -0.5:
#                 left_lines.append(x_bottom)
#     if left_lines:
#         left_lane_x = max(0, min(left_lines) - 50)
#     if right_lines:
#         right_lane_x = min(width, max(right_lines) + 50)
#     return int(left_lane_x), int(right_lane_x)

def get_ego_lane_bounds(lane_lines, width, height):
    if not lane_lines:
        return 0, width, None, None

    left_lane_x = width
    right_lane_x = 0
    left_lines = []
    right_lines = []
    left_line_points = []
    right_line_points = []

    for x1, y1, x2, y2 in lane_lines:
        if y1 != y2:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)

            if slope > 0.5:
                right_lines.append(x_bottom)
                right_line_points.append((x1, y1, x2, y2))
            elif slope < -0.5:
                left_lines.append(x_bottom)
                left_line_points.append((x1, y1, x2, y2))

    if left_lines:
        left_lane_x = max(0, min(left_lines) - 50)
    if right_lines:
        right_lane_x = min(width, max(right_lines) + 50)

    return (int(left_lane_x),
            int(right_lane_x),
            left_line_points if left_line_points else None,
            right_line_points if right_line_points else None)

def draw_lanes(frame, lane_lines):
    if lane_lines:
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lanes


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2
    xi1 = max(x1, x1_other)
    yi1 = max(y1, y1_other)
    xi2 = min(x2, x2_other)
    yi2 = min(y2, y2_other)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_other - x1_other) * (y2_other - y1_other)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

@app.route("/")
def index():
    return render_template("index.html")

def handle_video_upload(new_video_path):
    """Handle video upload by replacing the current video"""
    current_video_path = os.path.join(UPLOAD_FOLDER, CURRENT_VIDEO_FILE)
    try:
        # Remove existing video if it exists
        if os.path.exists(current_video_path):
            os.remove(current_video_path)
            logger.info(f"Removed existing video: {current_video_path}")
        
        # Move the new video to the standard name
        shutil.move(new_video_path, current_video_path)
        logger.info(f"New video saved as: {current_video_path}")
        return True
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        return False

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV files only.'}), 400

    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

    try:
        # Save uploaded file temporarily
        video.save(temp_filepath)

        # Verify the file was saved and is not empty
        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            return jsonify({'error': 'Failed to save video file'}), 500

        # Move it and rename it to final name
        if handle_video_upload(temp_filepath):
            return jsonify({
                'video_id': CURRENT_VIDEO_FILE,
                'message': 'Video uploaded successfully'
            })
        else:
            return jsonify({'error': 'Failed to process video'}), 500
    except Exception as e:
        logger.error(f"Exception during upload: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp file: {cleanup_error}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# Modify the video_feed route to use the current video
@app.route("/video_feed")
def video_feed():
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], CURRENT_VIDEO_FILE)
    if not os.path.exists(video_path):
        return jsonify({'error': 'No video available'}), 404
        
    logger.info(f"Starting video feed for: {video_path}")
    return Response(process_video(video_path),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return

    FPS = cap.get(cv2.CAP_PROP_FPS)
    FRAME_TIME = 1 / FPS
    frame_count = 0
    prev_frame = None
    prev_tracks = {}

    try:
        with app.app_context():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logger.info("End of video stream reached")
                    break

                frame_count += 1
                if frame_count % 2 == 0:
                    continue

                frame = cv2.resize(frame, (640, 480))
                height, width = frame.shape[:2]

                # Draw ego lane
                lane_lines = detect_lanes(frame)
                left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)

                for x1, y1, x2, y2 in lane_lines:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw ego lane bounds
                cv2.line(frame, (left_lane_x, height), (left_lane_x, int(height * 0.6)), (0, 0, 255), 2)
                cv2.line(frame, (right_lane_x, height), (right_lane_x, int(height * 0.6)), (0, 0, 255), 2)

                gps = get_gps_data()
                ego_speed = gps["speed"]
                motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
                prev_frame = frame.copy()

                ego_speed_gps = calculate_speed_from_gps(ego_gps_history, gps["latitude"], gps["longitude"], frame_count, FRAME_TIME)
                draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

                results = model(frame)[0]
                detections = [[int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3])]
                              for b in results.boxes if int(b.cls[0]) in [2, 3, 5, 7]]
                tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

                test_data = []
                for t in tracked_objects:
                    if len(t) >= 5:
                        x1, y1, x2, y2, tid = map(int, t)
                        cx = (x1 + x2) // 2
                        cy = y2
                        w = x2 - x1
                        h = y2 - y1
                        dist = height - y2
                        in_lane = 1 if left_lane_x <= cx <= right_lane_x else 0
                        rel_x = (cx - left_lane_x) / (right_lane_x - left_lane_x) if right_lane_x > left_lane_x else 0.5
                        test_data.append([x1, y1, x2, y2, cx, cy, w, h, dist, in_lane, rel_x])

                predictions = frontier_clf.predict(test_data) if test_data else []
                frontier_idx = np.argmax(predictions) if np.any(predictions) else -1
                frontier_vehicle = tracked_objects[frontier_idx] if 0 <= frontier_idx < len(tracked_objects) else None

                for t in tracked_objects:
                    if len(t) < 5:
                        continue
                    x1, y1, x2, y2, tid = map(int, t)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    color = (255, 0, 0)
                    event_type = "Tracked"
                    motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status
                    ttc, frontier_speed = None, 0
                    is_critical_event = False

                    if np.array_equal(t, frontier_vehicle):
                        color = (0, 255, 0)
                        event_type = "Frontier"
                        frontier_speed = estimate_frontier_speed(tid, cy, frame_count, FRAME_TIME)
                        distance = height - y2
                        ttc = calculate_ttc(ego_speed, frontier_speed, distance)
                        if ttc < 2:
                            event_type = "Near Collision"
                        pred_x, pred_y = kalman_tracker.predict_next_position(cx, cy)
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

                        # Anomaly detection
                        features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
                        scaled = scaler.transform(features)
                        if anomaly_model.predict(scaled)[0] == -1:
                            event_type += " - Anomaly"

                        # Motion check
                        if tid in prev_tracks:
                            dx = np.linalg.norm(np.subtract((cx, cy), prev_tracks[tid]))
                            if dx < 0.5:
                                motion = "Sudden Stop Detected!"
                            elif dx > 5.0:
                                motion = "Harsh Braking"
                        prev_tracks[tid] = (cx, cy)

                        # Collision detection
                        is_collision = any(
                            calculate_iou([x1, y1, x2, y2], [int(o[0]), int(o[1]), int(o[2]), int(o[3])]) > 0.5
                            for o in tracked_objects if not np.array_equal(o, t) and len(o) >= 5
                        )
                        if is_collision:
                            motion = "Collided"

                        is_critical_event = motion in ["Collided", "Sudden Stop Detected!", "Harsh Braking"] or "Anomaly" in event_type or "Collision" in event_type

                    # Labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fs = 0.4
                    th = 1
                    pad = 3
                    spacing = 15
                    motion_text = f"Motion: {motion}"
                    speed_text = f"Speed: {frontier_speed:.1f} km/h" if frontier_speed else "Speed: N/A"
                    ttc_text = f"TTC: {ttc:.1f}s" if ttc and ttc != float('inf') else "TTC: N/A"
                    id_text = f"ID: {tid}"


                    labels = [motion_text, speed_text, ttc_text, id_text]
                    label_positions = [(x1, y1 - 80 + spacing * i) for i in range(len(labels))]

                    overlay = frame.copy()
                    for i, (text, pos) in enumerate(zip(labels, label_positions)):
                        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
                        bg_pos1 = (pos[0] - pad, pos[1] - tht - pad)
                        bg_pos2 = (pos[0] + tw + pad, pos[1] + pad)
                        bg_color = (0, 0, 255) if is_critical_event else (0, 0, 0)
                        cv2.rectangle(overlay, bg_pos1, bg_pos2, bg_color, -1)
                        cv2.putText(overlay, text, pos, font, fs, (255, 255, 255), th)

                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Log and emit
                    event = EventLog(vehicle_id=tid, event_type=event_type,
                                     x1=x1, y1=y1, x2=x2, y2=y2, ttc=None if ttc == float("inf") else ttc,
                                     latitude=gps["latitude"], longitude=gps["longitude"],
                                     motion_status=motion)
                    if is_critical_event:
                        db.session.add(event)

                    try:
                        event_data = {
                            "id": event.id,
                            "vehicle_id": tid,
                            "event_type": event_type,
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "ttc": "N/A" if not ttc or ttc == float("inf") else round(ttc, 2),
                            "latitude": current_gps["latitude"], "longitude": current_gps["longitude"],
                            "motion_status": motion,
                            "is_critical": is_critical_event
                        }
                        socketio.emit("new_event", event_data)
                    except Exception as e:
                        logger.error(f"Socket emit error: {e}")

                if frame_count % 30 == 0:
                    try:
                        db.session.commit()
                    except Exception as e:
                        logger.error(f"Commit failed: {e}")

                success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if success:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        logger.error(f"process_video crashed: {e}")
    finally:
        cap.release()
        with app.app_context():
            db.session.remove()


@app.route("/events", methods=["GET"])
def get_events():
    events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
    logger.info(f"Events retrieved: {len(events)}")
    data = [{
        "id": e.id,
        "vehicle_id": e.vehicle_id,
        "event_type": e.event_type,
        "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "N/A",
        "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
        "ttc": "N/A" if e.ttc is None or e.ttc == float('inf') else e.ttc,
        "latitude": e.latitude,
        "longitude": e.longitude,
        "motion_status": e.motion_status
    } for e in events]
    logger.debug(f"Data returned: {data}")
    return jsonify(data)

@app.route("/export_critical_events", methods=["GET"])
def export_critical_events():
    try:
        with app.app_context():
            critical_event_types = ["Collided", "Harsh Braking", "Sudden Stop Detected!"]
            critical_events = EventLog.query.filter(EventLog.motion_status.in_(critical_event_types)).all()

            if not critical_events:
                logger.info("No critical events found.")
                return jsonify({"error": "No critical events found."}), 404

            data = []
            for event in critical_events:
                data.append({
                    "ID": event.id,
                    "Vehicle ID": event.vehicle_id,
                    "Event Type": event.event_type,
                    "Motion Status": event.motion_status,
                    "Timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "N/A",
                    "X1": event.x1, "Y1": event.y1, "X2": event.x2, "Y2": event.y2,
                    "TTC (s)": "N/A" if event.ttc is None or event.ttc == float('inf') else event.ttc,
                    "Latitude": event.latitude,
                    "Longitude": event.longitude
                })

            df = pd.DataFrame(data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Critical Events")

            # Define the new directory for blockchain excels
            BLOCKCHAIN_EXCELS_DIR = os.path.join(os.path.dirname(app.instance_path), "blockchain_excles")
            os.makedirs(BLOCKCHAIN_EXCELS_DIR, exist_ok=True)

            # Save the file to the new directory
            filename = f"critical_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = os.path.join(BLOCKCHAIN_EXCELS_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(output.getvalue())

            logger.info(f"Critical events exported to {filepath}")
            return jsonify({"message": "Critical events exported successfully to blockchain_excles folder."})

    except Exception as e:
        logger.error(f"Error exporting critical events: {e}")
        return jsonify({"error": "Failed to export critical events."}), 500

@app.route("/list_exported_files", methods=["GET"])
def list_exported_files():
    try:
        files = []
        for idx, filename in enumerate(os.listdir(EXPORT_DIR)):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(EXPORT_DIR, filename)
                timestamp = os.path.getmtime(file_path)
                files.append({
                    "id": idx + 1,
                    "file_name": filename,
                    "timestamp": int(timestamp * 1000)
                })
        logger.info(f"Listed {len(files)} exported files")
        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error listing exported files: {e}")
        return jsonify({"error": "Failed to list exported files"}), 500

@app.route("/delete_exported_file", methods=["POST"])
def delete_exported_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            logger.error("No filename provided")
            return jsonify({"error": "No filename provided"}), 400

        file_path = os.path.join(EXPORT_DIR, filename)
        if os.path.exists(file_path) and filename.endswith('.xlsx'):
            os.remove(file_path)
            logger.info(f"File deleted: {filename}")
            return jsonify({"message": "File deleted successfully"})
        else:
            logger.error(f"File not found or invalid: {filename}")
            return jsonify({"error": "File not found or invalid"}), 404
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"error": "Failed to delete file"}), 500

@app.route("/clear_exported_files", methods=["POST"])
def clear_exported_files():
    try:
        for filename in os.listdir(EXPORT_DIR):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(EXPORT_DIR, filename)
                os.remove(file_path)
        logger.info("All exported files cleared")
        return jsonify({"message": "All exported files cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing exported files: {e}")
        return jsonify({"error": "Failed to clear exported files"}), 500

@socketio.on('start_processing')
def handle_video_processing(data):
    video_id = data.get('video_id')
    if not video_id:
        socketio.emit('error', {'message': 'No video ID provided'})
        return
    
    # TODO: Implement actual video processing
    # This is a placeholder for the video processing logic
    # You'll need to integrate your existing video processing code here
    
    # Example of sending a frame
    socketio.emit('frame', {
        'frame': 'base64_encoded_frame_data'  # Replace with actual frame data
    })
    
    # Example of sending an event
    socketio.emit('event', {
        'type': 'Vehicle Detected',
        'description': 'A vehicle was detected in the frame',
        'timestamp': datetime.now().isoformat()
    })
    
    # Example of sending GPS data
    socketio.emit('gps_update', {
        'latitude': 37.7749,
        'longitude': -122.4194
    })
    
    # Start the actual video processing in a background thread
    threading.Thread(target=process_video, args=(video_id,)).start()

# def get_gps_data():
#     global current_latitude, current_longitude, gps_connected
#     if gps_connected:
#         # Calculate speed based on changes in latitude and longitude over time
#         # This is a simplified approach and may not be perfectly accurate
#         # A more robust approach would involve using timestamps from the GPS module
        
#         # For now, returning a placeholder speed or calculating based on rate of change
#         # Need to implement speed calculation from sequential GPS readings if needed for frontend display
        
#         # Returning current lat/lon and a placeholder speed
#         return {
#             "latitude": current_latitude,
#             "longitude": current_longitude,
#             "speed": 40.0, # Placeholder speed (km/h) - Update with calculated speed if needed
#             "connected": True
#         }
#     else:
#         return {
#             "latitude": 0.0,
#             "longitude": 0.0,
#             "speed": 0.0,
#             "connected": False
#         }


def get_gps_data():
    with gps_lock:
        return {
            "latitude": gps_data["latitude"],
            "longitude": gps_data["longitude"],
            "speed": 40.0 if gps_data["connected"] else 0.0,  # Placeholder
            "connected": gps_data["connected"]
        }


def process_live_stream(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {camera_id}")
        return

    FPS = 30  # Assuming 30 FPS for live stream
    FRAME_TIME = 1 / FPS
    prev_frame = None
    frame_count = 0
    prev_tracks = {}

    logger.info(f"Camera opened: FPS={FPS}, FRAME_TIME={FRAME_TIME}")

    try:
        with app.app_context():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame from camera")
                    break

                frame_count += 1
                if frame_count % 2 == 0:  # Process every other frame to reduce load
                    continue

                frame = cv2.resize(frame, (640, 480))
                height, width, _ = frame.shape
                center_x = width // 2

                # Lane detection
                lane_lines = detect_lanes(frame)
                left_lane_x, right_lane_x = get_ego_lane_bounds(lane_lines, width, height)
                draw_lanes(frame, lane_lines)
                for x1, y1, x2, y2 in lane_lines:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get GPS data
                current_gps = get_gps_data()
                lat = current_gps["latitude"]
                lon = current_gps["longitude"]
                ego_speed = current_gps["speed"]
                motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
                prev_frame = frame.copy()

                # Calculate ego vehicle speed using GPS data
                lat, lon = current_gps["latitude"], current_gps["longitude"]
                ego_speed_gps = calculate_speed_from_gps(ego_gps_history, lat, lon, frame_count, FRAME_TIME)

                # Draw speedometer
                draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

                # Object detection and tracking
                results = model(frame, verbose=False)[0]
                detections = []
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > 0.5:  # Confidence threshold
                        detections.append([x1, y1, x2, y2, score])

                if len(detections) > 0:
                    tracked_objects = tracker.update(np.array(detections))
                    frontier_vehicle = None
                    min_distance = float('inf')

                    for track in tracked_objects:
                        if len(track) < 5:
                            continue
                        x1, y1, x2, y2, track_id = map(int, track)
                        color = (255, 0, 0)
                        event_type = "Tracked"
                        ttc = None
                        vehicle_motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status

                        # Calculate distance to ego vehicle
                        distance = height - y2
                        if distance < min_distance:
                            min_distance = distance
                            frontier_vehicle = track

                        # Check for critical events
                        if np.array_equal(track, frontier_vehicle):
                            color = (0, 255, 0)
                            event_type = "Frontier"
                            y_center = (y1 + y2) // 2
                            frontier_speed = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME)
                            ttc = calculate_ttc(ego_speed, frontier_speed, distance) if frontier_speed and ego_speed else float('inf')
                            
                            if ttc < 2:
                                event_type = "Near Collision"
                            
                            x_center = (x1 + x2) // 2
                            pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
                            cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

                            # Anomaly detection
                            features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
                            scaled_features_array = scaler.transform(features)
                            scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
                            if anomaly_model.predict(scaled_features)[0] == -1:
                                event_type = f"{event_type} - Anomaly"

                        # Draw bounding box and labels
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add event to database if critical
                        is_critical_event = vehicle_motion in ["Collided", "Harsh Braking", "Sudden Stop Detected!"] or event_type == "Near Collision" or "Anomaly" in event_type;

                        if is_critical_event:
                            event = EventLog(
                                vehicle_id=track_id,
                                event_type=event_type,
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                ttc=ttc,
                                latitude=current_gps["latitude"],
                                longitude=current_gps["longitude"],
                                motion_status=vehicle_motion
                            )
                            db.session.add(event)
                            
                            # Emit event through WebSocket
                            event_data = {
                                "id": event.id,
                                "vehicle_id": event.vehicle_id,
                                "event_type": event_type,
                                "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "ttc": "N/A" if ttc is None or ttc == float('inf') else ttc,
                                "latitude": current_gps["latitude"],
                                "longitude": current_gps["longitude"],
                                "motion_status": vehicle_motion,
                                "is_critical": is_critical_event
                            }
                            socketio.emit('new_event', event_data)

                # Commit database changes periodically
                if frame_count % 30 == 0:
                    try:
                        db.session.commit()
                    except Exception as e:
                        logger.error(f"Commit failed: {e}")

                # Encode and send frame
                success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if success:
                    frame_bytes = buffer.tobytes()
                    socketio.emit('frame', {'frame': frame_bytes})

    except Exception as e:
        logger.error(f"Error in process_live_stream: {e}")
    finally:
        cap.release()
        with app.app_context():
            db.session.remove()

@socketio.on('start_live_processing')
def handle_live_processing(data):
    device_id = data.get('deviceId')
    if not device_id:
        socketio.emit('error', {'message': 'No device ID provided'})
        return
    
    # Start live stream processing in a background thread
    threading.Thread(target=process_live_stream, args=(device_id,)).start()

@socketio.on('stop_live_processing')
def handle_stop_live_processing():
    # The processing will stop when the camera is released
    pass

def cleanup_old_videos():
    try:
        video_path = os.path.join(UPLOAD_FOLDER, CURRENT_VIDEO_FILE)
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Old video deleted: {video_path}")
    except Exception as e:
        logger.error(f"Error during video cleanup: {e}")


# Add cleanup task to run periodically
def start_cleanup_task():
    while True:
        cleanup_old_videos()
        time.sleep(VIDEO_CLEANUP_INTERVAL)

# Start cleanup task in a background thread
cleanup_thread = threading.Thread(target=start_cleanup_task)
cleanup_thread.daemon = True
cleanup_thread.start()

__all__ = ['app', 'socketio', 'read_gps_data_from_serial']

if __name__ == "__main__":
    # Start the serial reading thread
    gps_thread = threading.Thread(target=read_gps_data_from_serial, args=(SERIAL_PORT_NAME, BAUDRATE))
    gps_thread.daemon = True # Allow the main thread to exit even if the GPS thread is running
    gps_thread.start()

    socketio.run(app, debug=True)



