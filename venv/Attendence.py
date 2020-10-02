import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import datetime
# from time import sleep


path = 'D:\Image_processing_projects\Face_Recognition\Image_Attendence'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def Speech(name):
    engine = pyttsx3.init()
    engine.say(name)
    engine.runAndWait()


def markAttendence(name):
    with open('D:\Image_processing_projects\Face_Recognition\Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []


        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name in nameList:
            name1 = 'Attendence Taken'
            cv2.putText(img,name1,(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            Speech(name1+name)
        elif name not in nameList:
            name2 = "Thank you For Enrolling"
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            cv2.putText(img, name2, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # sleep(2.0)
            Speech(name2+name)

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame= face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
            # Speech(name)

    cv2.imshow('Webcam',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
