import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# cl is classes/images iterating through the images directory (path)
for cl in myList:
    # reading the images in the path
    curImg = cv2.imread(f'{path}/{cl}')
    # appending image to images
    images.append(curImg)
    # removing the .jpg by splitting the text and grabbing the first element (the name)
    classNames.append(os.path.splitext(cl)[0])

# encoding images
def findEncodings(images):
    # creating the list
    encodeList = []
    # iterating through known and labeled persons
    for img in images:
        # cvt color to fix cv2 import color shift
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # encoding first element since it is a single image. face_encodings reads as img object
        encode = face_recognition.face_encodings(img)[0]
        # adding encoding to list
        encodeList.append(encode)
    # returning the List as the function output
    return encodeList

def markAttendance(name):
    # r+ allows read/write
    with open('Attendance.csv','r+') as f:
        # setting up the list using Attendance.csv aka f (still displaying the time)
        myDataList = f.readlines()
        # list of names
        nameList = []
        # iterating through each line, splitting the entries by , and appending the first entry (the name) to nameList
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # name is not already present? grab the time and format it, and write in a new line the name and formatted time
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Creating List with known encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# grabbing video feed
cap = cv2.VideoCapture(0)

while True:
    # loading in video feed, resizing, and converting color
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # identifying all faces in current frame
    facesCurFrame = face_recognition.face_locations(imgS)

    # encoding based on faces identified
    # Doing this based off of locations first allows for identification of multiple persons
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    #
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # drawing the box
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # displaying name
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            # calling mark attendance
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)