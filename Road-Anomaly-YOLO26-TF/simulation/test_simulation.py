import cv2
import numpy as np
import time
import os
import csv
import configparser
from datetime import datetime
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

# --- CONFIGURATION ---
config = configparser.ConfigParser()
config.read("simulation/detector.config")

MODEL_PATH = config.get("CONFIGURATION", "MODEL_PATH")
VIDEO_PATH = config.get("CONFIGURATION", "VIDEO_PATH")

INPUT_SIZE = config.getint("CONFIGURATION", "INPUT_SIZE")
CONF_THRESHOLD = config.getfloat("CONFIGURATION", "CONF_THRESHOLD")
NMS_THRESHOLD = config.getfloat("CONFIGURATION", "NMS_THRESHOLD")
LOG_COOLDOWN = config.getfloat("CONFIGURATION", "LOG_COOLDOWN")
THREADS = config.getint("CONFIGURATION", "THREADS")

# LOGGING SETUP 
VIDEO_STEM = Path(VIDEO_PATH).stem                  
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")   
LOG_FILE = Path("Logs") / f"{VIDEO_STEM}_{RUN_ID}.csv"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  

print(f" Log will be saved to: {LOG_FILE}")
print(f" Loading Model: {MODEL_PATH}")

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]

    print(" Model Loaded!")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

print(f" Opening Video: {VIDEO_PATH}")
print(input_details[0])
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_MSMF)

if not cap.isOpened():
    print(" Error: Could not open video file.")
    print("Check path:", os.path.abspath(VIDEO_PATH))
    exit()

print(" Video Opened! Press 'q' to quit.")

# Timer for log cooldown
last_log_time = 0

try:
    # Open CSV in write mode
    with LOG_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp", "Class", "Confidence"])

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            #A: PRE-PROCESSING (INT8 INPUT)
            display_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            input_data = np.expand_dims(display_img, axis=0).astype(np.float32) / 255.0
            #B: INFERENCE
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            #C: POST-PROCESSING
            output_data = np.squeeze(output_data).transpose()
            boxes, confidences = [], []

            for row in output_data:

                raw_conf = row[4]
                confidence = min(float(raw_conf), 1.0) 

                if confidence > CONF_THRESHOLD:
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    
                    x1 = int((cx - w/2) * INPUT_SIZE)
                    y1 = int((cy - h/2) * INPUT_SIZE)
                    width = int(w * INPUT_SIZE)
                    height = int(h * INPUT_SIZE)
                    
                    x1 = max(0, min(x1, INPUT_SIZE - 1))
                    y1 = max(0, min(y1, INPUT_SIZE - 1))
                    width = min(width, INPUT_SIZE - x1)
                    height = min(height, INPUT_SIZE - y1)
                    
                    boxes.append([x1, y1, width, height])
                    confidences.append(confidence)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

            if len(indices) > 0:
                current_time = time.time()
                # Check if enough time has passed since last log
                should_log = (current_time - last_log_time) > LOG_COOLDOWN
                
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]

                    # Draw Box (Visual feedback)
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(display_img, f"{int(conf*100)}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Log to CSV
                    if should_log:
                        csv_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Pothole",
                            f"{conf:.2f}"
                        ])
                        csv_file.flush()
                
                if should_log:
                    last_log_time = current_time

            # FPS Calculation
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            
            try:
                cv2.imshow("Road Anomaly Detector", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass # Continue running silently if no screen is found

except Exception as e:
    print(f" Runtime Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f" Simulation Finished. Log saved to: {LOG_FILE}")