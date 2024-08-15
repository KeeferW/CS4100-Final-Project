# FPS Game Computer Vision Project

This project analyzes aiming accuracy, recoil control, and shot detection in FPS games using computer vision techniques. The project utilizes OpenCV, YOLO, and Tesseract OCR for processing video footage of gameplay.

## Table of Contents

- [Project Overview](#project-overview)
- [Files and Structure](#files-and-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is designed to help players analyze their performance in first-person shooter (FPS) games by detecting and analyzing shots fired, crosshair placement, and recoil control. The project includes several scripts for different aspects of this analysis:

- `trackBullet.py`: Detects when a bullet is fired based on the ammo count displayed in the game.
- `isCrossHairProper.py`: Analyzes crosshair placement and determines whether it was properly aimed at a target at the moment of firing.
- `yolo_interface.py`: Trains and tests YOLO models for object detection tasks within the game footage.

## Files and Structure

### `trackBullet.py`

This script contains the following key functions:

- `open_video_file()`: Opens a file dialog to select a video file for analysis.
- `select_bounding_box(video_path)`: Allows the user to select the region of interest (ROI) in the video where the ammo count is displayed.
- `detect_ammo_drops(video_path, bbox)`: Analyzes the selected ROI to detect when bullets are fired by monitoring changes in the ammo count.
- `overlay_shots_fired(video_path, shots_fired, output_path)`: Overlays the number of shots fired on the original video and saves the output.

### `isCrossHairProper.py`

This script focuses on evaluating the player's crosshair placement:

- `is_crosshair_on_target(crosshair, bbox, frame)`: Determines whether the crosshair was properly aimed at a target's head or body at the time of a shot.
- Other utility functions handle frame-by-frame analysis and provide visual feedback on the accuracy of the player's aim.

### `yolo_interface.py`

This script is used for training and testing YOLO models:

- `model.train(data='ProjectPredictiveComputerVision/Project/Valorant Object Detection.v22i.yolov8/data.yaml', epochs=50, imgsz=640)`: Trains the YOLO model using the specified dataset.

## Setup Instructions

### Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.7 or higher
- OpenCV
- PyTorch and YOLOv8 (via the `ultralytics` package)
- Tesseract OCR

### Installing Dependencies


3. **Tesseract Installation**:
   - Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` path in `trackBullet.py` to match your installation directory.

### Setting Up YOLO

- Ensure that the YOLO model files (`models/test3.pt`, etc.) are correctly placed in the `models` directory.
- Update paths in `yolo_interface.py` as necessary.






## Contributing

This was a group effort of kaamil Thobani Keefer Wu Jon Blind

