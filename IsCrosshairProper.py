import cv2
import numpy as np
from ultralytics import YOLO
import os

# Log the current working directory
print("Current working directory:", os.getcwd())

# Import the bullet tracker
from trackBullet import detect_number_drop, select_bounding_box, open_video_file

# Load the object detection model
print("Loading YOLO model...")
model = YOLO('models/test3.pt')
print("Model loaded successfully.")

def is_crosshair_on_target(crosshair, bbox, frame):
    (x, y, w, h) = bbox
    # Define the head region as the top 20% of the detected bounding box height
    #head_height = h * 0.2  # 20% of the total bounding box height
    head_region = (x, y, w, head_height)  # Head region at the top of the bounding box

    crosshair_x, crosshair_y = crosshair

    # Draw the head region on the frame for debugging
    cv2.rectangle(frame, (int(head_region[0]), int(head_region[1])),
                  (int(head_region[0] + head_region[2]), int(head_region[1] + head_region[3])),
                  (0, 255, 0), 2)  # Green rectangle for head region

    # Check if the crosshair is within the head region
    if head_region[0] <= crosshair_x <= head_region[0] + head_region[2] and \
       head_region[1] <= crosshair_y <= head_region[1] + head_region[3]:
        return True
    return False

def save_frame_as_png(frame, frame_num, label):
    """Save the frame as a PNG file."""
    filename = f"shotsrun6-frame_{frame_num}_{label}.png"
    cv2.imwrite(filename, frame)
    print(f"Saved {label} frame at {frame_num} as {filename}")

# Load the video
video_path = open_video_file()
print("Video file selected:", video_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Get the bounding box for the ammo counter
bbox, first_frame = select_bounding_box(video_path)
if bbox is None:
    print("Error: No bounding box selected, exiting.")
    cap.release()
    exit()

print("Bounding box selected:", bbox)

# Detect the frames where bullets are fired
print("Detecting bullet drops...")
shot_frames = detect_number_drop(video_path, bbox, skip_frames=0)
print("Bullet fired at frames:", shot_frames)

# Get the dimensions of the video frame to find the center (crosshair position)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crosshair_position = (frame_width // 2, frame_height // 2)  # Center of the screen

print("Frame dimensions:", frame_width, "x", frame_height)
print("Crosshair position:", crosshair_position)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output-video-v13.mp4', fourcc, 20.0, (frame_width, frame_height))
print("Output video file initialized.")

frame_num = 0
buffer = []
buffer_size = 8  # Number of frames before and after the shot to buffer

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error in reading frame.")
        break

    # Add the current frame to the buffer
    buffer.append((frame_num, frame))

    # Process buffered frames for the shot if the current frame is within the shot range
    if frame_num in shot_frames:
        start_frame = max(frame_num - buffer_size, 0)
        end_frame = min(frame_num + buffer_size, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

        # Process all frames within the buffer range
        for buf_frame_num, buf_frame in buffer:
            if start_frame <= buf_frame_num <= end_frame:
                results = model(buf_frame)  # Perform inference

                for result in results:  # Iterate over each result in the list
                    if result.boxes:  # Check if there are any detections
                        for box in result.boxes:
                            bbox = box.xyxy[0].cpu().numpy()  # Extract the bounding box as a numpy array
                            x1, y1, x2, y2 = bbox
                            w = x2 - x1
                            h = y2 - y1

                            # Draw the full bounding box
                            cv2.rectangle(buf_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                            # Define the head region as the top 20% of the bounding box
                            head_height = h * 0.2
                            head_region = (x1+15, y1, w*.5, head_height)

                            # Draw the head region
                            cv2.rectangle(buf_frame, (int(head_region[0]), int(head_region[1])),
                                          (int(head_region[0] + head_region[2]), int(head_region[1] + head_region[3])),
                                          (0, 255, 0), 2)  # Green rectangle for head region

                            if is_crosshair_on_target(crosshair_position, head_region, buf_frame):
                                cv2.putText(buf_frame, "Headshot!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                save_frame_as_png(buf_frame, buf_frame_num, "headshot")
                            else:
                                # Calculate direction of miss
                                direction = "Center"
                                if crosshair_position[1] < y1:  # Crosshair is above
                                    direction = "Move Down"
                                elif crosshair_position[1] > y2:  # Crosshair is below
                                    direction = "Move Up"
                                elif crosshair_position[0] < x1:  # Crosshair is left
                                    direction = "Move Right"
                                elif crosshair_position[0] > x2:  # Crosshair is right
                                    direction = "Move Left"

                                cv2.putText(buf_frame, f"Missed! {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                save_frame_as_png(buf_frame, buf_frame_num, "missed")

                # Write the processed buffered frame to the output video
                out.write(buf_frame)

    # Write the unprocessed frame if it's not part of the buffered shot frames
    if frame_num not in shot_frames:
        out.write(frame)

    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing completed.")
