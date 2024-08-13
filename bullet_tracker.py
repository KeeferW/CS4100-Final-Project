import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Change this to wherever tesseract is installed, but this is the default im pretty sure, can also delete this and it'll still work probably

def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read video")
        cap.release()
        return None
    
    roi = cv2.selectROI("Select Ammo Count", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Ammo Count")
    
    cap.release()
    return roi

# Process video and detect ammo drops
def detect_ammo_drops(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_count = None

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop ROI
        x, y, w, h = roi
        ammo_region = frame[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(ammo_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        
        # Tesseract algorithm to recognize digits
        config = "--psm 7"
        ammo_count = pytesseract.image_to_string(gray, config=config, lang='eng')
        ammo_count = ''.join(filter(str.isdigit, ammo_count)) 
    
        if ammo_count.isdigit():
            current_count = int(ammo_count)
            
            if prev_count is not None and current_count < prev_count:
                timestamp = frame_number / fps
                print(f"Bullet fired at frame {frame_number}, timestamp {timestamp:.2f} seconds")
            
            prev_count = current_count
        elif prev_count is not None:
            pass

        frame_number += 1
    
    cap.release()

def main():
    video_path = "valorant example footage.mp4" 
    roi = select_roi(video_path)
    
    if roi is not None:
        detect_ammo_drops(video_path, roi)

if __name__ == "__main__":
    main()

