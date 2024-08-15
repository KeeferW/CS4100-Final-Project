import cv2
import numpy as np
from scipy.ndimage import label

# Function to select bounding box
def select_bounding_box(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read the video.")
        cap.release()
        return None, None
    
    bbox = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select Region")
    return bbox, frame

# Function to detect numbers within the ROI
def detect_number(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_imgs = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            digit_roi = thresh[y:y+h, x:x+w]
            digit_imgs.append(digit_roi)
    
    digit_imgs = sorted(digit_imgs, key=lambda img: cv2.boundingRect(img)[0])
    return len(digit_imgs) if digit_imgs else None

# Function to detect when the number drops by one
def detect_number_drop(video_path, bbox, skip_frames=1000):
    cap = cv2.VideoCapture(video_path)
    prev_number = None
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    shot_frames = []

    # Skip the first few frames
    while frame_num < skip_frames:
        ret, _ = cap.read()
        if not ret:
            print(f"Failed to skip frames. Only {frame_num} frames available.")
            cap.release()
            return []
        frame_num += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop to the bounding box
        x, y, w, h = bbox
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Detect the number in the current frame
        current_number = detect_number(cropped_frame)
        
        if current_number is not None and prev_number is not None:
            if current_number == prev_number - 1:
                shot_frames.append(frame_num)
        
        prev_number = current_number
        frame_num += 1
    
    cap.release()
    return shot_frames


def open_video_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    return video_path
