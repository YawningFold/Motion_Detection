import cv2
import imutils
import time

cam=cv2.VideoCapture(0)
time.sleep(1)

first_frame=None
a=500

while True:
    f,img=cam.read()
    txt="No Motion Detected"
    img=imutils.resize(img,width=500)

    g_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian_img=cv2.GaussianBlur(g_img,(21,21),0)

    if first_frame is None:
        first_frame=gaussian_img
        continue
    img_diff=cv2.absdiff(first_frame, gaussian_img)

    __,thresh_img=cv2.threshold(img_diff,25,255,cv2.THRESH_BINARY)
    thresh_img=cv2.dilate(thresh_img,None,iterations=3)

    cnts=cv2.findContours(thresh_img.copy(),cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<a:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        txt="Mothion Detected"
    
    cv2.putText(img,txt,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("Camera",img)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



