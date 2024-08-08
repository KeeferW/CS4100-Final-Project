import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import label

# This file accepts an .mp4, and allows the user to draw a box around the ammo counter. Whenever the number drops by 1 (a shot is fired), it prints the frame and timestamp.

# Function to select bounding box
def select_bounding_box(video_path):
    # Open MP4
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read the video.")
        cap.release()
        return None, None
    
    # Select a window on the first frame (Draw a box around the ammo)
    bbox = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select Region")
    return bbox, frame

# Function to detect numbers within the ROI
def detect_number(roi):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_imgs = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            digit_roi = thresh[y:y+h, x:x+w]
            digit_imgs.append(digit_roi)
    
    # Sort digits by their x position (left to right)
    digit_imgs = sorted(digit_imgs, key=lambda img: cv2.boundingRect(img)[0])
    
    # Count the number of digits detected
    return len(digit_imgs) if digit_imgs else None

# Function to detect when the number drops by one
def detect_number_drop(video_path, bbox):
    cap = cv2.VideoCapture(video_path)
    prev_number = None
    
    frame_num = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp for current frame
        timestamp = frame_num / fps
        
        # Crop to the bounding box
        x, y, w, h = bbox
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Detect the number in the current frame
        current_number = detect_number(cropped_frame)
        
        if current_number is not None and prev_number is not None:
            if current_number == prev_number - 1:
                print(f"Number dropped from {prev_number} to {current_number} at frame: {frame_num}, time: {timestamp:.2f} seconds")
        
        prev_number = current_number
        frame_num += 1
    
    cap.release()

def open_video_file():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    return video_path

if __name__ == "__main__":
    video_path = open_video_file()
    if not video_path:
        print("No video file selected.")
    else:
        bbox, first_frame = select_bounding_box(video_path)
        if bbox is not None:
            detect_number_drop(video_path, bbox)
