import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'runs/detect/train10/weights/best.pt'
CAMERA_ID = 3
WINDOW_NAME = "YOLOv8 Fullscreen Detection" # We'll use this name

# --- SCRIPT ---
model = YOLO(MODEL_PATH)

# Connect to the virtual webcam
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"❌ Error: Could not open camera with ID {CAMERA_ID}.")
    exit()

# --- ADD THESE TWO LINES FOR FULLSCREEN ---
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🚀 Connected to DroidCam. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Use the same window name here
        cv2.imshow(WINDOW_NAME, annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()