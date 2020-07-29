import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry import Polygon

path = 'ImagesAttendance'
adminPath = 'ImagesAdmin'
images = []
adminImages = []
adminNames = []
classNames = []
boxId = 0
boxes = []
boxesList = np.array([[]])
myList = os.listdir(path)
myAdminList = os.listdir(adminPath)
print(myList)

# cl is classes/images iterating through the images directory (path)
for cl in myList:
    # reading the images in the path
    curImg = cv2.imread(f'{path}/{cl}')
    # appending image to images
    images.append(curImg)
    # removing the .jpg by splitting the text and grabbing the first element (the name)
    classNames.append(os.path.splitext(cl)[0])

for cl in myAdminList:
    curImg = cv2.imread(f'{adminPath}/{cl}')
    adminImages.append(curImg)
    adminNames.append(os.path.splitext(cl)[0])

# encoding images
def findEncodings(images):
    # creating the list
    encodeList = []
    # iterating through input faces
    for img in images:
        # cvt color to fix cv2 import color shift
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # encoding first element since it is a single image. face_encodings reads as img object
        encode = face_recognition.face_encodings(img)[0]
        # adding encoding to list
        encodeList.append(encode)
    # returning the List as the function output
    return encodeList

def findAdminEncodings(images):
    # creating the list
    adminEncodeList = []
    # iterating through input faces
    for img in images:
        # cvt color to fix cv2 import color shift
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # encoding first element since it is a single image. face_encodings reads as img object
        encode = face_recognition.face_encodings(img)[0]
        # adding encoding to list
        adminEncodeList.append(encode)
    # returning the List as the function output
    return adminEncodeList

def markAttendance(name):
    # r+ allows read/write
    with open('Attendance.csv','r+') as f:
        # setting up the list using Attendance.csv aka f
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
adminList = findAdminEncodings(adminImages)
print('Encoding Complete')

# grabbing video feed
cap = cv2.VideoCapture(0)

# number of boxes to divide the screen into
# needs to be a square number
boxesAmt = 9
displayGrid=False
showLine=False
while True:
    # loading in video feed, resizing, and converting color
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgHeight, imgWidth, channels = img.shape
    imgMiddle = int(imgWidth/2), int(imgHeight/2)
    # identifying all faces in current frame
    facesCurFrame = face_recognition.face_locations(imgS)

    # drawing quadrants
    # dividing image into boxes
    M = imgHeight//3
    N = imgWidth//3
    if displayGrid:
        for y in range(0, imgHeight, M):
            for x in range(0, imgWidth, N):
                y1 = y + M
                x1 = x + N
                tiles = img[y:y+M,x:x+N]
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0))
                if len(boxes) < boxesAmt:
                    boxes.append([x,y,x1,y1])
            for box in boxes:
                print(box)

    # encoding based on faces identified
    # Doing this based off of locations first allows for identification of multiple persons
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    # grabbing the encoded face [0] and the location of the face [1] in combined (zipped) lists
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        # comparing the faces from the known list to the current encoded face
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        # calculating distance (value comparing likeness, lower is closer to true)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchesAdmin = face_recognition.compare_faces(adminList, encodeFace)
        faceDisAdmin = face_recognition.face_distance(adminList, encodeFace)
        adminMatchIndex = np.argmin(faceDisAdmin)
        print(faceDis)
        print(faceDisAdmin)
        # Indexes the distances so we know which person we're referencing
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # drawing the box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(114,70,133),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(114,70,133),cv2.FILLED)
            # displaying name
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            # calling mark attendance
            markAttendance(name)
        elif matchesAdmin[adminMatchIndex]:
            name = adminNames[adminMatchIndex].upper()
            # drawing the box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            faceMiddle = int((x1+x2)/2), int((y1+y2)/2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
          #  cv2.rectangle(img, (x1, y2 + 35), (x2, y2), (0, 255, 0), cv2.FILLED)


            lineColor = (255,255,255)
            # drawing a line to the middle of admin face
            if showLine:
                cv2.line(img, imgMiddle , faceMiddle, lineColor, 2)

            # displaying name
           # cv2.putText(img, 'ADMIN', (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            # unknown person
            cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
