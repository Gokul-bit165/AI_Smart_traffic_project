# --- ADD THESE TWO LINES AT THE VERY TOP ---
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# -----------------------------------------

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pre-trained model to start from
    model = YOLO("yolov8n.pt")

    # Train the model, forcing it to use a single worker
    model.train(data="dataset/data.yaml", epochs=50, imgsz=640, workers=0)

    print("🎉🎉🎉 Custom model training complete! 🎉🎉🎉")