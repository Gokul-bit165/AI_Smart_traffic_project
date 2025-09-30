import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import serial
import time
from ultralytics import YOLO

VIDEO_PATH = 'VID20250927035305.mp4' 

MODEL_PATH = 'runs/detect/train10/weights/best.pt'
ESP32_PORT = 'COM17' # Set to your ESP32's COM port

# Class IDs
CAR_CLASS = 0
AMBULANCE_CLASS = 1

# --- INITIALIZATION ---
try:
    ser = serial.Serial(ESP32_PORT, 115200, timeout=1)
    print(f"✅ Connected to ESP32 on {ESP32_PORT}")
except Exception as e:
    print(f"❌ Failed to connect to ESP32: {e}")
    ser = None

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video file: {VIDEO_PATH}")
    exit()

print("🚀 Processing video. Press 'q' to quit.")

# Timer for sending commands periodically
last_command_time = 0
COMMAND_INTERVAL = 1.0 # Send commands to ESP32 every 1 second

# --- MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        print("Video finished. Looping...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to the beginning
        continue

    frame = cv2.resize(frame, (1280, 720))
    
    # --- DYNAMICALLY CREATE A MAXIMIZED ROI FOR LANE 2 ---
    height, width, _ = frame.shape
    y1 = 200
    y2 = 650
    LANE2_ROI = [0, y1, width, y2]

    # --- Detection and Counting in LANE 2 ROI ---
    car_count = 0
    ambulance_detected = False
    
    x1_roi, y1_roi, x2_roi, y2_roi = LANE2_ROI
    roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]
    results = model(roi_frame, verbose=False)

    for result in results:
        for box in result.boxes:
            b_x1, b_y1, b_x2, b_y2 = map(int, box.xyxy[0])
            start_point = (x1_roi + b_x1, y1_roi + b_y1)
            end_point = (x1_roi + b_x2, y1_roi + b_y2)
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            if class_id == AMBULANCE_CLASS:
                ambulance_detected = True
                color = (0, 0, 255) # Red for ambulance
            else:
                car_count += 1
                color = (255, 165, 0) # Orange for cars
            
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # --- Send Commands to ESP32 Periodically ---
    current_time = time.time()
    if current_time - last_command_time > COMMAND_INTERVAL:
        last_command_time = current_time
        if ser:
            if ambulance_detected:
                ser.write(b"AMBULANCE_LANE_2\n")
                print("Sent: AMBULANCE_LANE_2")
            else:
                command = f"COUNT_LANE_2={car_count}\n"
                ser.write(command.encode())
                print(f"Sent: {command.strip()}")
            
    # --- Visualization ---
    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 255), 3)
    cv2.putText(frame, "AI SENSOR ZONE (LANE 2)", (x1_roi + 10, y1_roi + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Detected Cars: {car_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.imshow("AI Video Processing", frame)
    
    # waitKey controls playback speed. Lower number = faster video.
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if ser: ser.close()