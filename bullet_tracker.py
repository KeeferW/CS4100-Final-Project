import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # You can replace this with wherever the tesseract library is located, but this is the default im p sure

def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read video")
        cap.release()
        return None
    
    roi = cv2.selectROI("Select Ammo Count", frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select Ammo Count")
    
    cap.release()
    return roi

def detect_ammo_drops(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_count = None
    shots_fired = []
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
                shots_fired.append((frame_number, current_count))
                timestamp = frame_number / fps
                print(f"Bullet fired at frame {frame_number}, timestamp {timestamp:.2f} seconds")
            
            prev_count = current_count

        frame_number += 1
    
    cap.release()
    return shots_fired

def overlay_shots_fired(video_path, shots_fired, output_path="output_with_counter.mp4"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    shots_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if shots_fired and frame_number == shots_fired[0][0]:
            shots_count += 1
            shots_fired.pop(0)

        # Overlay counter
        cv2.putText(frame, f'Shots Fired: {shots_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_number += 1
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

def main():
    video_path = "valorant.mp4"
    roi = select_roi(video_path)
    
    if roi is not None:
        shots_fired = detect_ammo_drops(video_path, roi)
        overlay_shots_fired(video_path, shots_fired)

if __name__ == "__main__":
    main()
