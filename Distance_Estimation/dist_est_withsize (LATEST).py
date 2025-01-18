import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from gtts import gTTS
from queue import Queue, Empty
import time
import threading
import speech_recognition as sr


# Global Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'
frame_rate = 30
alpha = 0.4
previous_depth = 0.0

# Load Models
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(device).eval()
yolo_model = YOLO('yolov8n.pt').to(device)
transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# Speech Queue
speech_queue = Queue(maxsize=1)
last_no_objects = 0
fallback = 5
last_audio_times = {}
audio_interval = 2  # Interval between audio feedback (in seconds)
detection_active = True  # Toggle for Start/Stop Detection
paused = False  # Toggle for Pause/Resume Detection

def listen_for_command():
    """Listen for voice commands."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Voice Command] Listening for a command...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize(audio).lower()
            print(f"[Voice Command] Heard: {command}")
            return command
        except Exception as e:
            print(f"[Voice Command] Error: {e}")
        return None

def handle_command(command):
    """Handle the recognized voice command."""
    global detection_active, paused, audio_interval

    if "start detection" in command:
        detection_active = True
        print("[Command] Detection started.")
    elif "close app" in command:
        detection_active = False
        print("[Command] Detection stopped.")
    elif "stop detection" in command:
        paused = True
        add_to_speech_queue("Detection paused. Say 'Resume detection' to continue.")
        print("[Command] Detection paused.")
    elif "resume detection" in command:
        paused = False
        print("[Command] Detection resumed.")
    elif "change alert frequency to" in command:
        try:
            new_interval = int(command.split("change alert frequency to")[1].strip().split()[0])
            audio_interval = new_interval
            print(f"[Command] Alert frequency changed to {audio_interval} seconds.")
        except ValueError:
            print("[Command] Invalid alert frequency command.")

def voice_input_thread():
    """Thread for continuously listening to voice commands."""
    while True:
        command = listen_for_command()
        if command:
            handle_command(command)


# Helper Functions
def apply_ema_filter(current_depth):
    """Apply exponential moving average for depth smoothing."""
    global previous_depth
    filtered_depth = alpha * current_depth 
    previous_depth = filtered_depth
    return filtered_depth

def depth_to_distance(depth_value, depth_scale=1.0):
    """Convert normalized depth value to approximate distance."""
    return 1.0 / (depth_value * depth_scale) if depth_value > 0 else float('inf')

def add_to_speech_queue(message):
    """Add a message to the speech queue."""
    try:
        if not speech_queue.empty():
            speech_queue.get_nowait()  # Remove existing message
        speech_queue.put_nowait(message)
    except Exception as e:
        print(f"[Queue Error]: {e}")

def speak():
    """Handle text-to-speech in a separate thread."""
    while True:
        try:
            message = speech_queue.get(timeout=0)
            if message:
                tts = gTTS(text=message, lang='en')
                tts.save("temp.mp3")
                os.system("mpg321 temp.mp3")  # Play audio
                os.remove("temp.mp3")
                print(f"[TTS] Speaking: {message}")
        except Empty:
            continue

def estimate_steps_to_avoid(bbox_center, filtered_distance, object_width_pixels, frame_width):
    """Estimate the number of steps and direction to avoid the object."""
    
    stride_length = 0.6  # Average human step length in meters
    safety_margin = 0.5
    object_width_meters = object_width_pixels / frame_width * filtered_distance
    print('Frame Width:',frame_width//2)
    print("bbox_center:",bbox_center)
    if bbox_center > (frame_width // 2) and bbox_center < (frame_width // 2) + 200:
        move_direction = "move left"
    elif bbox_center  > (frame_width // 2) - 200 and bbox_center  <= (frame_width // 2):
        move_direction = "move right"
    else:
        move_direction = ''
    
    required_clearance =  safety_margin + object_width_meters
    if filtered_distance > required_clearance:
        steps_to_avoid = ''  # No movement required if the distance is greater than required clearance
    else:
        steps_to_avoid = int(required_clearance / stride_length) + 1 # Calculate how many steps to move based on the stride length
    print(steps_to_avoid)
    return move_direction, steps_to_avoid
    

def process_frame(frame):
    """Process the video frame for depth estimation and object detection."""
    global last_audio_times
    global last_no_objects
    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Depth Estimation
    imgbatch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
        ).squeeze().cpu().numpy()
    output_norm = cv2.normalize(prediction, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Object Detection
    results = yolo_model(frame)
    frame_width = frame.shape[1]

    object_detected = False

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for idx, box in enumerate(boxes):
            if confidences[idx] > 0.7:
                x1, y1, x2, y2 = map(int, box)
                class_name = results[0].names[class_ids[idx]]

                # Estimate object distance
                mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                object_width_pixels = x2 - x1
                bbox_center = (x1 + x2) // 2
                direction = get_direction(bbox_center, frame_width)

                if 0 <= mid_x < output_norm.shape[1] and 0 <= mid_y < output_norm.shape[0]:
                    depth_value = output_norm[mid_y, mid_x]
                    distance = depth_to_distance(depth_value)
                    filtered_distance = apply_ema_filter(distance)

                    # Estimate steps and avoidance direction
                    move_direction, steps_to_avoid = estimate_steps_to_avoid(bbox_center, filtered_distance, object_width_pixels, frame_width)

                    # Draw bounding box and label
                    draw_bounding_box(frame, x1, y1, x2, y2, class_name, confidences[idx], filtered_distance)

                    # Prepare distance message    
                    distance_message = f"{class_name} is approximately {filtered_distance:.1f} meters {direction}. {move_direction}"
                    if steps_to_avoid and move_direction:
                        distance_message = f"{class_name} is approximately {filtered_distance:.1f} meters {direction}. {move_direction} {steps_to_avoid} steps"
                    object_detected = True

                    current_time = time.time()
                    if class_name not in last_audio_times or (current_time - last_audio_times[class_name] >= audio_interval):
                        try:
                            add_to_speech_queue(distance_message)
                            last_audio_times[class_name] = current_time
                        except Exception as e:
                            print(f"[Audio Error]: {e}")

    # Estimate distances for undetected regions
    if object_detected == False:
        mid_x, mid_y = frame.shape[1] // 2, frame.shape[0] // 2
        depth_value = output_norm[mid_y, mid_x]
        distance = depth_to_distance(depth_value)
        filtered_distance = apply_ema_filter(distance)
        undetected_message = f"Closest obstacle is possibily {filtered_distance:.1f} meters ahead. Move cautiously."
        current_time = time.time()
        if current_time - last_no_objects >= fallback:
            add_to_speech_queue(undetected_message)
            last_no_objects = current_time

    return frame

def get_direction(bbox_center, frame_width):
    """Determine the relative direction of an object."""
    if bbox_center > (frame_width // 2) + 120:
        return "to your right"
    elif bbox_center < (frame_width // 2) - 120:
        return "to your left"
    return ""

def draw_bounding_box(frame, x1, y1, x2, y2, class_name, confidence, distance):
    """Draw bounding box and label on the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"{class_name} {confidence:.2f} {distance:.1f}m", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_frame_with_state(frame):
    """Process frame based on detection state."""
    global detection_active, paused

    if paused:
        # If detection is paused, display a message and return the original frame
        cv2.putText(frame, "Detection paused. Say 'Resume detection' to continue.",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    # If detection is active, process the frame as usual
    return process_frame(frame)

# Main Function
def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    # Start speech thread
    threading.Thread(target=speak, daemon=True).start()
    threading.Thread(target=voice_input_thread, daemon=True).start()

    while detection_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping for consistent FPS
        current_time = time.time()
        if (current_time - prev_time) < 1.0 / frame_rate:
            continue
        prev_time = current_time

        frame = process_frame_with_state(frame)

        # Show the frame
        cv2.imshow("Real-time Object Detection & Depth Estimation", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
