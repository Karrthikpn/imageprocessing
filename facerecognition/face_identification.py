import os
import numpy as np 
from PIL import Image
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('/media/karthik/EC742F3F742F0C40/Friends_tech/gesture-recognition/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(BASE_DIR,"images")

current_id=0
label_ids={}

x_train=[]
y_train=[]


for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()

            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id +=1

            id_ =label_ids[label]
            #print(label_ids)

            #x_train.append(path)
            #y_train.append(label)

            pil_img=Image.open(path).convert('L')
            img_array=np.array(pil_img,"uint8")
            #print(img_array)
            
            faces=face_cascade.detectMultiScale(img_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = img_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_train.append(id_)

with open("labels.pickel","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_train))
recognizer.save("trainner1.yml")