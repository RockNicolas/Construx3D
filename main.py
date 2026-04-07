import cv2
import time

webcam = cv2.VideoCapture(0)

cv2.namedWindow("Video da Webcam", cv2.WINDOW_NORMAL)

if webcam.isOpened():
    p_time = 0
    
    while True:
        validacao, frame = webcam.read()
        if not validacao:
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video da Webcam", frame)
        
        key = cv2.waitKey(1)
        if key == 27: # ESC
            cv2.imwrite("FotoLira.png", frame)
            break

webcam.release()
cv2.destroyAllWindows()