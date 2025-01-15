import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import pyttsx3
from scipy.interpolate import RectBivariateSpline
import queue

# Load YOLOv8 model (pre-trained)
yolo_model = YOLO('yolov8s.pt') 

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Load MiDaS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
midas.to('cpu')
midas.eval()
engine = pyttsx3.init()
# Apply transforms for MiDaS
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

# Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

# Define depth to distance
def depth_to_distance(depth_value,depth_scale):
    return 1.0 / (depth_value*depth_scale)

# Start video capture (from camera)
cap = cv2.VideoCapture(0)  # 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to RGB for MediaPipe and YOLO
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects with YOLOv8
    results = yolo_model(frame)  # Get results from YOLOv8
    # Extract bounding boxes and class information
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
    classes = results[0].names # Get object class names
    
    # Process the frame with MediaPipe Pose
    pose_results = pose.process(img)

    if pose_results.pose_landmarks:
        # Extract pose landmarks
        landmarks = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
        
        waist_landmarks = [pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                           pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]]
        mid_point = ((waist_landmarks[0].x + waist_landmarks[1].x) / 2,
                     (waist_landmarks[0].y + waist_landmarks[1].y) / 2,
                     (waist_landmarks[0].z + waist_landmarks[1].z) / 2)
        mid_x, mid_y,mid_z = mid_point

        imgbatch = transform(img).to('cpu')

        # MiDaS Depth Estimation
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Create a spline for depth interpolation
        h, w = output_norm.shape
        x_grid = np.arange(w)
        y_grid = np.arange(h)
        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_mid_filt = spline(mid_y, mid_x)
        depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
        depth_mid_filt = apply_ema_filter(depth_midas) / 10  # Apply EMA and scale

        # Display the depth estimation

        # Draw object detections on the frame
        for idx,box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = int(results[0].boxes.cls[idx].item())  # Class index (integer)
            class_name = results[0].names[class_id]  # Class name from the YOLOv8 model
    # Get the confidence score for this bounding box
            confidence = results[0].boxes.conf[idx].item() 
            
            if confidence > 0.8:  # Only consider detections with confidence > 50%
                color = (0, 255, 0)  # Green for detected objects
    
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Class: + {class_name}" + "Depth in units: " + str(np.format_float_positional(depth_mid_filt , precision=3)), 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                '''cv2.putText(frame, f"{class_name} {np.format_float_positional(depth_mid_filt , precision=3):.2f} {confidence}m", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''
                distance_message = class_name +" is approximately " + str(np.format_float_positional(depth_mid_filt , precision=2)) + " meters away."
                engine.say(distance_message)
                engine.runAndWait()
        # Show the frame
        cv2.imshow('Real-time Object Detection & Depth Estimation', frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
