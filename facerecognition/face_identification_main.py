import cv2
import numpy as np 
import pickle

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('/media/karthik/EC742F3F742F0C40/Friends_tech/gesture-recognition/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner1.yml")


labels={}
with open("labels.pickel","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
while True:
    _,frame=cap.read() #camera will start reading 

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #to convert color video to gray

    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 : #and conf<=85
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,0,255)
            stroke=4
            cv2.putText(frame,name,(x,y),font,3,color,stroke)


        item_gray="my_img.png"

        color=(255,0,0)
        stroke=4
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

        cv2.imwrite(item_gray,roi_gray)

    
    cv2.imshow("video frames",frame)  #to display frames

    if cv2.waitKey(1) & 0xff==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

