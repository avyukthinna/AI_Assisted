import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
#import pyttsx3
from gtts import gTTS
from queue import Queue, Empty
from scipy.interpolate import RectBivariateSpline
import time
import threading

# Initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(device).eval()
yolo_model = YOLO('yolov8n.pt').to(device)
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
frame_rate = 30
prev_time = 0

# Depth smoothing
alpha = 0.4
previous_depth = 0.0

last_audio_time = 0  # Track the time of the last audio feedback
audio_interval = 7  # Set the interval between audio feedback (in seconds)

# Queue for passing distance messages to the speech thread
speech_queue = Queue()

def apply_ema_filter(current_depth):
    """Apply exponential moving average for depth smoothing."""
    global previous_depth
    filtered_depth = alpha * current_depth #+ (1 - alpha) * previous_depth
    previous_depth = filtered_depth
    return filtered_depth

def depth_to_distance(depth_value, depth_scale=1.0):
    """Convert normalized depth value to approximate distance."""
    return 1.0 / (depth_value * depth_scale) if depth_value > 0 else float('inf')

def speak():
    """Function to handle text-to-speech in a separate thread using gTTS."""
    while True:
        try:
            # Wait for a message from the main loop
            message = speech_queue.get(timeout=1)  # Wait for a maximum of 1 second for a message
            if message:
                tts = gTTS(text=message, lang='en')
                tts.save("temp.mp3")  # Save the audio to a temporary file
                os.system("mpg321 temp.mp3")  # Play the audio using mpg321 (you can also use another player)
                os.remove("temp.mp3")  # Remove the temporary file after playing
                print(f"[TTS] Speaking: {message}")
        except Empty:
            continue  # No message to speak, continue waiting

# Start speech thread
speech_thread = threading.Thread(target=speak, daemon=True)
speech_thread.start()

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame skipping for consistent FPS
    current_time = time.time()
    if (current_time - prev_time) < 1.0 / frame_rate:
        continue
    prev_time = current_time

    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Depth estimation with MiDaS
    imgbatch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
        ).squeeze().cpu().numpy()

    output_norm = cv2.normalize(prediction, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
   
    # YOLOv8 Object Detection
    results = yolo_model(frame)
    frame_width = frame.shape[1]
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
        for idx, box in enumerate(boxes):
            confidence = confidences[idx]
            
            if confidence > 0.5:  # Filter low-confidence detections
                x1, y1, x2, y2 = map(int, box)
                class_name = results[0].names[class_ids[idx]]

                # Estimate object distance using the depth map
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)
                bbox_center = (x1 + x2) // 2
                direction = "to your right" if bbox_center > frame_width // 2 else "to your left"
                if 0 <= mid_x < output_norm.shape[1] and 0 <= mid_y < output_norm.shape[0]:
                    depth_value = output_norm[mid_y, mid_x]
                    distance = depth_to_distance(depth_value)
                    filtered_distance = apply_ema_filter(distance)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name} {filtered_distance:.1f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Prepare distance message for speech thread
                    distance_message = f"{class_name} is approximately {filtered_distance:.1f} meters {direction}"
                    if current_time - last_audio_time >= audio_interval:
                        try:
                            # Put the message in the speech queue for the speech thread to handle
                            speech_queue.put(distance_message)
                            print(speech_queue)
                            last_audio_time = current_time
                        except Exception as e:
                            print(f"[Audio Error]: {e}")

    # Show the frame
    cv2.imshow("Real-time Object Detection & Depth Estimation", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
