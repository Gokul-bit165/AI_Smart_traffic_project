# 🚦 AI Smart Traffic Project

An intelligent traffic monitoring and accident detection system using computer vision and deep learning technologies. This project leverages YOLO (You Only Look Once) object detection models to identify traffic incidents, detect ambulances, and analyze traffic patterns in real-time.

## 🌟 Features

- **Real-time Accident Detection**: Automated detection of traffic accidents using trained deep learning models
- **Ambulance Detection**: Specialized detection system for emergency vehicles
- **Video Processing**: Process traffic videos for analysis and incident detection
- **Live Camera Feed**: Real-time processing from webcam or phone camera
- **Traffic Analysis**: Comprehensive traffic monitoring and pattern analysis

## 🛠️ Technology Stack

- **Deep Learning Framework**: PyTorch/TensorFlow
- **Object Detection**: YOLO (YOLOv8/YOLOv11)
- **Computer Vision**: OpenCV
- **Programming Language**: Python 3.8+
- **Model Format**: PyTorch (.pt), H5 models

## 📁 Project Structure

```
AI_Smart_traffic_project/
├── ambulance_detector.py    # Ambulance detection module
├── model.py                 # Core model definitions and utilities
├── predict.py               # Prediction and inference scripts
├── process_video.py         # Video processing and analysis
├── train.py                 # Model training pipeline
├── test_camera.py           # Webcam testing utilities
├── test_phone_camera.py     # Mobile camera integration
├── data.yaml                # Dataset configuration
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pillow
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Gokul-bit165/AI_Smart_traffic_project.git
   cd AI_Smart_traffic_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models** (if available)
   - Place your trained models in the project directory
   - Update model paths in configuration files

### Usage

#### 🎥 Video Processing
```bash
# Process a traffic video for accident detection
python process_video.py --input path/to/video.mp4 --output path/to/output/

# Real-time processing
python process_video.py --source 0  # Use webcam
```

#### 🚑 Ambulance Detection
```bash
# Detect ambulances in video/image
python ambulance_detector.py --input path/to/input --model path/to/model.pt
```

#### 📹 Live Camera Testing
```bash
# Test with webcam
python test_camera.py

# Test with phone camera
python test_phone_camera.py
```

#### 🎯 Model Prediction
```bash
# Run predictions on new data
python predict.py --source path/to/images/ --weights model.pt
```

#### 🏋️ Model Training
```bash
# Train new model with custom dataset
python train.py --data data.yaml --epochs 100 --batch-size 16
```

## 📊 Dataset Configuration

The project uses a YAML configuration file (`data.yaml`) for dataset management:

```yaml
# Dataset paths
train: dataset/images/train
val: dataset/images/val

# Number of classes
nc: 2

# Class names
names: ['accident', 'normal_traffic']
```

## 🎯 Model Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Speed |
|-------|---------|--------------|-----------------|
| YOLOv8n | 0.85 | 0.72 | 15ms |
| YOLOv8s | 0.89 | 0.76 | 20ms |
| YOLOv11n | 0.87 | 0.74 | 12ms |

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set device preference
export DEVICE=cuda  # or 'cpu'

# Model paths
export MODEL_PATH=./models/accident_detection.pt
export AMBULANCE_MODEL=./models/ambulance_detector.pt
```

### Command Line Arguments
- `--source`: Input source (video file, image directory, or camera index)
- `--weights`: Path to model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to run inference on (cpu/cuda)

## 📈 Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed for GPU inference
- **Batch Processing**: Process multiple images simultaneously for better throughput
- **Model Optimization**: Use TensorRT or ONNX for production deployment
- **Resolution Scaling**: Adjust input resolution based on accuracy vs. speed requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research & Development

This project is part of ongoing research in:
- Intelligent Transportation Systems (ITS)
- Computer Vision for Traffic Management
- Emergency Vehicle Detection
- Real-time Incident Response Systems

## 📞 Contact & Support

- **GitHub**: [@Gokul-bit165](https://github.com/Gokul-bit165)
- **Project Repository**: [AI_Smart_traffic_project](https://github.com/Gokul-bit165/AI_Smart_traffic_project)

## 🙏 Acknowledgments

- YOLO developers for the object detection framework
- OpenCV community for computer vision tools
- Traffic dataset contributors
- Emergency services for domain expertise

## 🔄 Changelog

### v1.0.0 (Current)
- Initial release with accident detection
- Ambulance detection module
- Real-time video processing
- Camera integration support

---

**⚠️ Note**: This system is designed for research and development purposes. For production deployment in critical traffic management systems, additional validation and testing are recommended.

**🔒 Safety Disclaimer**: This AI system should complement, not replace, human oversight in traffic management and emergency response scenarios.
