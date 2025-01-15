import sys
import time
import pyttsx3
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLO model if necessary

# Function to calculate the distance of an object in meters
def calculate_distance(bbox_width, focal_length=712.50, real_width=15):
    if bbox_width == 0:
        return None
    distance_cm = (real_width * focal_length) / bbox_width
    return round(distance_cm / 100, 2)  # Convert to meters

# Open webcam feed
cap = cv2.VideoCapture(0)  # 0 for default webcam, or replace with camera index

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

# Initialize variables for time tracking
last_update_time = time.time()
last_speech_time = time.time()
update_interval = 10
distance_smoothing_window = 5
distance_queue = []

# Process webcam frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    frame_width = frame.shape[1]  # Get the width of the frame

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            bbox_width = x2 - x1
            current_time = time.time()

            # Calculate direction (left or right)
            bbox_center = (x1 + x2) // 2
            direction = "to your right" if bbox_center > frame_width // 2 else "to your left"

            if current_time - last_update_time >= update_interval:
                distance = calculate_distance(bbox_width)
                last_update_time = current_time

                if distance is not None:
                    distance_queue.append(distance)
                    if len(distance_queue) > distance_smoothing_window:
                        distance_queue.pop(0)

                    last_distance = distance_queue[-1] if distance_queue else None
                    distance_text = f"Distance: {last_distance} m"

                    if last_distance is not None:
                        cv2.putText(frame, distance_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if last_distance < 2:
                        if current_time - last_speech_time >= update_interval:
                            speech_text = f"{label} is at {last_distance} meters {direction}"
                            engine.say(speech_text)
                            engine.runAndWait()
                            last_speech_time = current_time

    cv2.imshow("YOLO Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()