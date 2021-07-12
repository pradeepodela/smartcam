import numpy as np 
from tensorflow import keras
import tensorflow
from recorder import Recorder
import matplotlib.pyplot as plt
import scipy
import librosa
from librosa import display
from IPython.display import Audio
from scipy.fft import fft, fftfreq
import numpy as np
import cv2
import serial
import cvzone
import smtplib
import imghdr
from numpy.core.numeric import indices
from email.message import EmailMessage
from tracker import *



fpsReader = cvzone.FPS()
classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

thres = 0.5
nms_threshold = 0.2
classfile = r'D:\archive\Object_Detection_Files\coco.names'
configpath = r'D:\archive\Object_Detection_Files\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = r'D:\archive\Object_Detection_Files\frozen_inference_graph.pb'
tracker = EuclideanDistTracker()




net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)






data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('D:/dowanloadss/converted_keras (1)new/keras_model.h5')



amc = 0
fir = 0
pol = 0
def emergency_vechicledetecation(frame):
    
    frame = cv2.resize(frame,(224,224))
    img_res = np.asarray(frame)
    normalized_image_array = (img_res.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
   
    ambulance = (f'ambulance:{prediction[0][0].round()}')
    firengine = (f'firengine:{prediction[0][1].round()}')
    policecar = (f'policecar:{prediction[0][2].round()}')
    trafic =    (f'trafic:{prediction[0][3].round()}')
   
    text = (f'{ambulance}% {firengine}% {policecar}% {trafic}% {amc}')
   
    return text

def email(frame,mail):
    Sender_Email = "odelapradeep12@gmail.com"
    Reciever_Email = mail
    Password = 'pradeep9246'
    img = frame
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "rule volited" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('the following person voilated the trafic rule here is the image ') 

    with open(img, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        
        smtp.login(Sender_Email, Password)              
        smtp.send_message(newMessage)
    print('success msg went')


def img_p(frame , thres=0.6 , nms_threshold=0.2 , pltbox = False , tracker_plt = False ):

    if tracker_plt:
        pltbox = False
    

    nw_label = []
    obj = {}
    detections = []
    img = frame
    fps, img = fpsReader.update(img,pos=(50,80),color=(0,255,0),scale=5,thickness=4)
    
    print(f'fps={round(fps)}')



    classid , comfs , bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(comfs).reshape(1,-1)[0])
    confs = list(map(float,comfs))


    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)


    for i in indices:
        i = i[0]
        
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        label =  classnames[classid[i][0]-1]
        detections.append([x, y, w, h])
        nw_label.append(label)



        if pltbox:
            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            cv2.putText(img,label,(box[0]+5,box[1]+10),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
           
        
        ### for counting the obj
        for n in nw_label:
            obj[n] = nw_label.count(n)

            ### object tracking
    if tracker_plt:
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print(str(id))
            

    return img , obj
