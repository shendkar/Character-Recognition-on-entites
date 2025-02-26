import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time
import tensorflow as tf
import glob
from ctypes import *
import time
import math
import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, date
import sys
from PyQt5.QtWidgets import *
import PySimpleGUI as sg
import PIL
from PIL import Image, ImageTk
import io
import pandas as pd

def check(arr):
    if np.all(arr == 0):
        return True
    return False

def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}
            
def bubbleSort(arr1,arr2,arr3):
    n = len(arr1)
    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr1[j] > arr1[j + 1]:
                arr1[j], arr1[j + 1] = arr1[j + 1], arr1[j]
                arr2[j], arr2[j + 1] = arr2[j + 1], arr2[j]
                arr3[j], arr3[j + 1] = arr3[j + 1], arr3[j]
    return arr1,arr2,arr3

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x))
    xmax = int(round(x + w))
    ymin = int(round(y))
    ymax = int(round(y + h))
    return xmin, ymin, xmax, ymax

score_THRESHOLD = 0.65
NMS_THRESHOLD = 0.85
cmap = plt.get_cmap('tab20b')
COLORS = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

wid, hid = sg.Window.get_screen_size()
class_names = []
im = r'imagestest'
path = "imagestest"
cfg_path = r'yolov4-custom.cfg'
weights_path = r'yolov4-custom_best.weights'
classes_path = r'new_obj.names'

sg.theme('DarkBrown1')
layout = 	[
	[sg.Text('Forged Entity Character Recognizer', size=(150,2), font=('Any',50),text_color='#1c86ee' ,justification='center')],
	[sg.Text('', size=(150,2), font=('Any',50),text_color='#1c86ee' ,justification='center')],
	[sg.Text('', font=('Any',15))],
	[sg.Text('Path to input images', font=('Any',15)), sg.In(im,size=(150,40), font=('Any',15), key='input'), sg.FileBrowse(font=('Any',15))],
	[sg.Text('YOLO cfg Path', font=('Any',15)), sg.In(cfg_path,size=(150,40), font=('Any',15), key='config_file'), sg.FolderBrowse(font=('Any',15))],
	[sg.Text('YOLO weight Path', font=('Any',15)), sg.In(weights_path,size=(150,40), font=('Any',15), key='weights'), sg.FolderBrowse(font=('Any',15))],
	[sg.Text('YOLO Classes Path', font=('Any',15)), sg.In(classes_path,size=(150,40), font=('Any',15), key='classes'), sg.FolderBrowse(font=('Any',15))],
        [sg.Text('Threshold', font=('Any',15)), sg.Slider(range=(0,100), orientation='h', resolution=1, default_value=65, size=(40,40), font=('Any',15), key='thresh'),sg.Text('NMS Threshold', font=('Any',15)), sg.Slider(range=(0,100), orientation='h', resolution=1, default_value=85, size=(40,40), font=('Any',15), key='nmsthresh'),sg.OK(font=('Any',15)),sg.Cancel(font=('Any',15))]	
	]
win = sg.Window('YOLO Image', layout, resizable=True ,
			default_element_size=(50,2),
			text_justification='center',
			auto_size_text=True).Finalize()
win.Maximize()
event, values = win.read()
if event is None or event =='Cancel':
 exit()
args = values
score_THRESHOLD = args['thresh']/100
NMS_THRESHOLD = args['nmsthresh']/100
with open(args['classes']) as f:
 class_names = [cname.strip() for cname in f.readlines()]
net = cv2.dnn.readNet(args['weights'], args['config_file'])
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
path = args['input']
dict = [['Name','Label Detected','Label and its Confidence','Date/Time']]
df = pd.DataFrame(dict)
df.to_csv('Labelsdetected.csv', mode='w', header = False, index =False)
for filename in glob.iglob(path + '**/*.jpg', recursive=True):
   image = cv2.imread(filename)
   roi = None
   cv2.namedWindow('ROI Selector', cv2.WND_PROP_FULLSCREEN)
   cv2.setWindowProperty('ROI Selector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
   roi = cv2.selectROI('ROI Selector', image)
   roi1 = np.asarray(roi)
   if roi is not None and check(roi1) is False:
    cv2.destroyWindow('ROI Selector')
    image2 = np.zeros([image.shape[0], image.shape[1], 3], np.uint8)
    roi_cropped = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    image2[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = roi_cropped
    classes, scores, boxes = model.detect(image2, score_THRESHOLD, NMS_THRESHOLD)
    win.Close()
    all_detection = []
    lefts = []
    confidences = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        class_name = "%s" % (class_names[classid])
        classid = "%s" % (class_names[classid])
        colors = class_colors(classid)
        cv2.rectangle(image2, box, (255,255,255), 2)
        w = box[3] - box[1]
        h = box[2] - box[0]
        left, top, right, bottom = bbox2points(box)
        cv2.rectangle(image2, (left,top), (right,bottom), colors[classid], 2)
        cv2.putText(image2, "{}".format(classid),(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 5, colors[classid], 16, cv2.LINE_AA)
        all_detection.append(classid)
        confidences.append(score)
        lefts.append(left)
    lefts, all_detection,confidences = bubbleSort(lefts, all_detection,confidences)
    detss=""
    for ele in all_detection: 
     detss += ele
    cv2.putText(image2, "{}".format(detss), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 16, cv2.LINE_AA)
    com_det_con = []
    #for spacing in the CSV files of the predicted labels some special characters are given for clean formatting
    for i in range(len(all_detection)):
     com_det_con.append("{}:-{:.2f}".format(all_detection[i],confidences[i]*100))
    dict = [filename]
    now = datetime.now()
    dict.append(all_detection)
    dict.append(com_det_con) #For printing confidence scores
    dict.append(now) #Printing Date/Time Information on running the algo on the image.
    #Saving the image xlsx file containing Detections, Confidence and current Date/time
    combined = []
    #Modified on June 8, 2022
    #dict = {'Name':[image_name], 'Label Detected': [all_detection], 'Detection Confidences': [com_det_con], 'Date/Time': [now]}
    dict = [[filename, all_detection,com_det_con,now]]
    col_names = ['Name','Label Detected','com_det_con','Date/Time']
    df = pd.DataFrame(dict)
    df.to_csv('Labelsdetected.csv', mode='a', header = False, index =False)
    df1 = pd.read_csv('Labelsdetected.csv')
    df1.to_excel('Labelsdetected.xlsx', index =False)
    cv2.imwrite('pred.jpg',image2)
    cv2.resize(image2,(1280,720))
    cv2.namedWindow('YOLOv4', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('YOLOv4',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('YOLOv4',image2)
    cv2.waitKey(0)
