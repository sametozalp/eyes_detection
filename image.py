import cv2

img = cv2.imread("face-detection/face.png")

face_cascade = cv2.CascadeClassifier("cascade/frontalface.xml")
eyes_cascade = cv2.CascadeClassifier("cascade/eye.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5) # ölçeklendirme ve pencere

for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
    
img2 = img[y:y+h, x:x+w]
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

eyes = eyes_cascade.detectMultiScale(gray2)

for ex,ey,ew,eh in eyes:
    cv2.rectangle(img2, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
    
cv2.imshow("image", img)
cv2.waitKey(0)