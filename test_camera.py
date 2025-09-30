import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'runs/detect/train10/weights/best.pt'

# Change the Camera ID to 0 for the built-in webcam
CAMERA_ID = 1

WINDOW_NAME = "YOLOv8 Built-in Cam Detection"

# --- SCRIPT ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"❌ Error: Could not open built-in camera with ID {CAMERA_ID}.")
    print("Please check if it is being used by another application or is disabled.")
    exit()

# Optional: Create a fullscreen window
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🚀 Starting built-in camera feed. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Feed closed.")