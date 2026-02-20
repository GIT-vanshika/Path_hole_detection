Edge-Based Pothole Detection System
A real-time computer vision solution designed to detect road surface anomalies using lightweight deep learning models. Optimized for edge computing devices such as the Raspberry Pi, this system performs high-frequency inference while maintaining data integrity and minimizing hardware resource consumption.

Project Overview
This system leverages a custom-trained YOLO neural network, optimized via quantization to INT8 precision for TensorFlow Lite. It is engineered to operate autonomously in vehicular environments, detecting potholes and logging geospatial/temporal data for infrastructure maintenance analysis.

Key Features
Optimized Inference: distinct YOLO architecture converted to TFLite (INT8) for high-speed performance on ARM CPU architectures.
Data Integrity: Implements immediate buffer flushing to ensure zero data loss during sudden power interruptions common in automotive environments.
Resource Management: Features an intelligent logging cooldown system (1.0s interval) to prevent I/O saturation and extend SD card lifespan.
False Positive Suppression: Utilizes a strict confidence threshold (0.60) and signal clamping to ensure only high-certainty anomalies are recorded.
Autonomous Operation: Includes headless mode support, absolute path resolution, and camera retry logic for reliable startup without human intervention.
Technical Stack
Language: Python 3.12
Inference Engine: TensorFlow Lite Runtime (tflite_runtime)
Computer Vision: OpenCV (cv2)
Data Handling: NumPy, CSV
Directory Structure
text

/project_root
├── main.py                # Primary inference execution script
├── models/
│   └── best_model.tflite  # INT8 Quantized Model file
├── Logs/                  # Automatically generated CSV detection logs
└── simulation/            # Test clips for development
Setup and Usage
1. Environment Setup
Ensure the target device has Python 3 installed. For Raspberry Pi deployment, use the lightweight TFLite runtime to reduce overhead.

Bash

pip install tflite-runtime opencv-python-headless numpy
2. Model Deployment
Place the quantized .tflite model file into the models/ directory.

3. Execution
Run the main script. The system will automatically detect the camera, load the model, and begin logging.

Bash

python3 main.py
Operational Logic
Initialization: The system resolves absolute paths and attempts to initialize the video feed with a 5-attempt retry mechanism.
Inference: Frames are resized to 480x480 and processed. Output coordinates are clamped to the frame boundaries to prevent overflow errors.
Filtering: Detections below 60% confidence are discarded. Non-Maximum Suppression (NMS) removes duplicate bounding boxes.
Logging: Valid detections are written to a timestamped CSV file. The file buffer is flushed immediately after every write operation to guarantee persistence.

Follow These lines to download data:

!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="baZbg4YRiN1PBEMao7aY")
project = rf.workspace("smartathon").project("new-pothole-detection")
version = project.version(2)
dataset = version.download("yolo26")
