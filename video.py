import cv2

vid = cv2.VideoCapture("eyes_detection/eye.mp4")

face_cascade = cv2.CascadeClassifier("cascade/frontalface.xml")
eyes_cascade = cv2.CascadeClassifier("cascade/eye.xml")

while True:
    ret, frame = vid.read()
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
    eye_frame = frame[y:y+h, x:x+w]
    gray2 = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eyes_cascade.detectMultiScale(gray2)
    
    for ex,ey,ew,eh in eyes:
        cv2.rectangle(eye_frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
    
vid.release()
cv2.destroyAllWindows()