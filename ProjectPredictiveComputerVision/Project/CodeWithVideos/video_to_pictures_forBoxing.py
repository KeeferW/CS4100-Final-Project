import cv2
import time 
cpt = 0 
maxFrames = 2000
cap = cv2.VideoCapture('Project/valorantgameplay.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1020,500))
    cv2.imwrite('Project/video_to_pictures_forBoxingFolderv2/frame%d.jpg' %cpt, frame)
    cpt += 1
    print(cpt)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()