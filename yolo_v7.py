import cv2
from numpy import random
from collections import deque
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from byte_track.bytetracker import ByteTrack
from yolov7.yolov7_detector import YOLOv7Detector

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

global names
names = load_classes('data/escaleras.names')


colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
speed_four_line_queue = {}
object_counter = {}



def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 1: #No_barandal  #BGR
        color = (26,26,255)
    elif label == 0: # Barandal
        color = (29,215,0)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        
        cv2.line(img, c1, c2, color, 30)
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def estimateSpeed(location1, location2):

	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = 8 #Pixels per Meter 
	d_meters = d_pixels / ppm
	time_constant = 15 * 3.6
	speed = d_meters * time_constant
	return speed

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])






def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):
    # cv2.line(img, line2[0], line2[1], (0,200,0), 3)
    
    height, width, _ = img.shape 
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = x1 + int(box[2]), y1 + int(box[3])
        box=[x1, y1, x2, y2]

        # x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        
        # box_area = (x2-x1) * (y2-y1)
        box_height = (y2-y1)

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        
        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
          speed_four_line_queue[id] = [] 

        color = compute_color_for_labels(int(object_id[i]))
        obj_name = names[int(object_id[i])]
        label = '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)

        UI_box(box, img, label=label, color=color, line_thickness=2)
            

    count = 0
    for idx, (key, value) in enumerate(object_counter.items()):
        # print(idx, key, value)
        cnt_str = str(key) + ": " + str(value)

        cv2.line(img, (width - 150 ,25+ (idx*40)), (width,25 + (idx*40)), [85,45,255], 30)
        cv2.putText(img, cnt_str, (width - 150, 35 + (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        count += value

    return img, count

def load_yolov7_and_process_each_frame(model, vid_name, enable_GPU, save_video, confidence, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stframe):
    data_deque.clear()
    speed_four_line_queue.clear()
    object_counter.clear()

    if model == 'yolov7':
        weights = 'yolov7/weights/yolov7.onnx'
    elif model == 'yolov7-tiny':
        weights = 'yolov7/weights/yolov7-tiny.onnx'
    elif model == 'yolov7-escaleras':
        weights = 'yolov7/weights/escaleras_tiny.onnx'
    else:
        print('Model Not Found!')
        exit()

    detector = YOLOv7Detector(weights=weights, use_cuda=enable_GPU, use_onnx=True)
    tracker = ByteTrack(detector)
    # dataset = LoadImages(vid_name, img_size=1280, auto_size=64)

    vdo = cv2.VideoCapture(vid_name)
    results = []
    start = time.time()
    count = 0
    frame_id = 0
    prevTime = 0

    fourcc = 'mp4v'  # output video codec
    fps = vdo.get(cv2.CAP_PROP_FPS)
    w = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter('inference/output/results.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    while vdo.isOpened():
    # for path, img, im0s, vid_cap in dataset:
        curr_time = time.time()
        frame_id +=1
        _, img = vdo.read()
        
        if _ == False:
            break

        # img = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        # img, count, obj_ids = tracker.inference(img, conf_thresh=confidence, classes=assigned_class_id)
        bboxes, ids, scores, obj_ids = tracker.inference(img, conf_thresh=confidence, classes=assigned_class_id)
        # print(bboxes[0].shape if len(bboxes)>0 else None)
        img, count = draw_boxes(img, bboxes, obj_ids, identities=ids)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

                        # Save results (image with detections)
        cv2.line(img, (20,25), (127,25), [85,45,255], 30)
        cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        if save_video:
            vid_writer.write(img)

        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{fps:.1f}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{len(data_deque)}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{count}</h1>", unsafe_allow_html=True)
        # if frame_id%3==0:
        #     stframe.image(img, channels = 'BGR',use_column_width=True)
        stframe.image(img, channels = 'BGR',use_column_width=True)


    end = time.time()
    print('Done. (%.3fs)' % (end - start))
    cv2.destroyAllWindows()    
    vdo.release()
    vid_writer.release()