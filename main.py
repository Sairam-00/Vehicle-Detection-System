import cv2
import numpy as np
from time import sleep
import mysql.connector
from database import*


thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture("./video.mp4")

width_min=80 #Minimum rectangle width
height_min=80
delay=60
subtract_back = cv2.bgsegm.createBackgroundSubtractorMOG()

def get_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cars=0
Totalcount=0
tcc=0
unc=0
cc=0
mc=0
bc=0
tc=0

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = './ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = './frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

count_line=450

while True:
    c1=0
    c2=0
    detect=[]
    det=[]
    _ ,img = cap.read()
    img2=img.copy()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    cv2.line(img, (25, count_line), (1200, count_line), (255,127,0), 3)
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in indices:
        index = i[0]
        box = bbox[index]
        x,y,w,h = box[0],box[1],box[2],box[3]
        centre=get_centre(x,y,w,h)
        label=classNames[classIds[index][0]-1]
        confidence=str(round(confs[index],2))
        detect.append([centre,label,confidence])
        cv2.circle(img, centre, 4, (0, 0,255), -1)
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,label +""+confidence,(x,y+20),font,1,(0,255,0),2)

    offset=4
    for i in detect:
        if i[0][1]<(count_line+offset) and i[0][1]>(count_line-offset):
            cars+=1
            c1=c1+1
            cls=i[1]
            if cls == 'car':
                cc=cc+1
                a=str(cc)
                upd(a,"3")
            elif cls=='motorbike':
                mc=mc+1
                a=str(mc)
                upd(str(a),"4")
            elif cls=='bus':
                bc=bc+1
                a=str(bc)
                upd(str(a),"5")
            elif cls=='truck':
                tc=tc+1
                a=str(tc)
                upd(str(a),"6")
            print("vehicle is detected :"+str(cars),cls,i[2])
            detect.remove(i)

    temp = float(1/delay)
    sleep(temp)
    grey = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract_back.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    dilated_ = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(dilated_,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.line(img, (25, count_line), (1200, count_line), (255,127,0), 3)
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contour = (w >= width_min) and (h >= height_min)
        if not valid_contour:
            continue

        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        centre = get_centre(x, y, w, h)
        det.append(centre)
        #cv2.circle(frame, centre, 4, (0, 0,255), -1)

        for (x,y) in det:
            if y<556 and y>544:
                Totalcount+=1
                c2=c2+1
                tcc=tcc+1
                a=str(tcc)
                upd(str(a),"1")
                det.remove((x,y))
                print("car is detected : "+str(Totalcount))
    sol=abs(tcc-cars)
    upd(str(sol),"2")
    cv2.imshow("Output",img)
    key=cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
