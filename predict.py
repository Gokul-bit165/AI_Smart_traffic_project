import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import argparse
import serial # --- NEW IMPORT ---
import time   # --- NEW IMPORT ---

# =============================================================================
# 1. DEFINE CONSTANTS & PARAMETERS
# =============================================================================
IMG_HEIGHT = 128
IMG_WIDTH = 128
MAX_FRAMES = 20
MODEL_PATH = 'accident_detection_full_dataset_model.h5'

# --- NEW: SERIAL COMMUNICATION SETUP ---
# IMPORTANT: Replace '/dev/ttyS3' with your ESP32's port name (e.g., COM3 in Windows -> /dev/ttyS3 in WSL)
ESP32_PORT = 'COM17'
BAUD_RATE = 9600

# =============================================================================
# 2. LOAD MODEL & SETUP SERIAL CONNECTION
# =============================================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# --- NEW: ESTABLISH SERIAL CONNECTION ---
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for the connection to establish
    print(f"Successfully connected to ESP32 on {ESP32_PORT}")
except serial.SerialException as e:
    print(f"Error: Could not connect to ESP32 on {ESP32_PORT}. Please check the port name.")
    print(e)
    ser = None # Set to None so the script can run without a connection for testing

# =============================================================================
# 3. SET UP VIDEO INFERENCE
# =============================================================================
parser = argparse.ArgumentParser(description='Accident Detection from Video')
parser.add_argument('--video', type=str, required=True, help='Path to the test video file.')
args = parser.parse_args()

frame_buffer = deque(maxlen=MAX_FRAMES)
video_capture = cv2.VideoCapture(args.video)

frame_counter = 0
PREDICTION_INTERVAL = 10
prediction_text = "Status: Initializing..."
prediction_color = (0, 255, 0)

print("Starting video processing... Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_counter += 1
    processed_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized_frame = processed_frame / 255.0
    frame_buffer.append(normalized_frame)

    if len(frame_buffer) == MAX_FRAMES and frame_counter % PREDICTION_INTERVAL == 0:
        input_data = np.expand_dims(np.array(frame_buffer), axis=0)
        prediction = model.predict(input_data, verbose=0)[0][0]

        if prediction > 0.6:
            prediction_text = f"Status: ACCIDENT DETECTED! ({prediction:.2f})"
            prediction_color = (0, 0, 255)
            # --- NEW: SEND 'A' FOR ACCIDENT ---
            if ser: ser.write(b'A')
        else:
            prediction_text = f"Status: Normal ({prediction:.2f})"
            prediction_color = (0, 255, 0)
            # --- NEW: SEND 'N' FOR NORMAL ---
            if ser: ser.write(b'N')
    
    cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, prediction_color, 2)
    cv2.imshow('Real-time Accident Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
# --- NEW: CLOSE SERIAL PORT ---
if ser:
    ser.write(b'N') # Reset pin to Normal state on exit
    ser.close()
    print("Serial connection closed.")
print("Video processing finished.")