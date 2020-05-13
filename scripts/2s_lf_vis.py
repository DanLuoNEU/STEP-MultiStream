# Late fusion in a visualization way with RGB and OF results
# 04/28/2020, Dan, Yuexi and Tim

import os
import cv2
import csv
import h5py

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import fnmatch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


font = ImageFont.truetype("utils/arial.ttf", 28)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def draw_rectangle(draw, coords, outline, width=4):
    for i in range(int(width/2)):
        draw.rectangle(coords-i, outline=outline, fill=None)
        # cv2.rectangle()
    for i in range(1,int(width/2)):
        draw.rectangle(coords+i, outline=outline, fill=None)

def write_vis(pred_boxes,img_path,save_path,id2class,thd=0.6):

    I = Image.open(img_path)
    W,H=I.size
    draw = ImageDraw.Draw(I)

    if pred_boxes is not None:
        for key in pred_boxes:
            box = np.asarray(key.split(','), dtype=np.float32)
            box[::2] *= W
            box[1::2] *= H

            count = 0
            for label, score in pred_boxes[key]:
                if id2class[label] == "background":
                    continue
                if score < thd:
                    continue
                else:
                    draw_rectangle(draw, box, outline="red")
                    draw.text((box[0] + 10, box[1] + 10 + count * 20),
                          '{}'.format(id2class[label] if id2class is not None else label), fill="red", font=font)
                break
                count += 1
    I.save(save_path)
    print(save_path)

def main():
    rgb_path = 'examples/results/rgb_results.npz'
    of_path = 'examples/results/of_results.txt'
    img_path = 'examples/rgb_frames/cam20-p2p-2'
    out_path = 'examples/results/cam20-p2p-2-20200427-rgb-lf_v2'
    img_list = fnmatch.filter(os.listdir(img_path), '*.jpg')
    img_list.sort()
    id2class = {1:'transfer', 2: 'transfer', 3:'background'}
    # Read in RGB results
    rgb_results = np.load(rgb_path, allow_pickle=True)['allResult']
    # Read in OF results
    of_results = {} # keys: timesec, value:pred_boxes[keys]
    with open(of_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            key = row[2]+','+row[3]+','+row[4]+','+row[5]
            label = int(row[6])
            prob = float(row[7])
            if row[1] not in of_results.keys():
                of_results[row[1]] = {}
                of_results[row[1]][key] = []
                of_results[row[1]][key].append((label,prob))
            else:
                if key not in of_results[row[1]].keys():
                    of_results[row[1]][key] = []
                    of_results[row[1]][key].append((label,prob))
                else:
                    of_results[row[1]][key].append((label,prob))
    print('rgb: ', len(rgb_results), 'flow: ', len(of_results))
    # Loop the RGB bboxes based on time
    for t, pred_boxes in enumerate(rgb_results):
        if t==0:
            write_vis(pred_boxes,os.path.join(img_path,img_list[t]),os.path.join(out_path,img_list[t]),id2class=id2class)
            continue
        # print(t)
        if str(t+1).zfill(4) in of_results.keys():
            pred_boxes_of = of_results[str(t+1).zfill(4)] #optical flow files are 1 frame behind
        else:
            write_vis(pred_boxes,os.path.join(img_path,img_list[t]),os.path.join(out_path,img_list[t]),id2class=id2class)
            continue
        # Loop RGB bboxes
        for cod_key in pred_boxes:
            # Compute the IOU and choose the largest from of results
            cod_rgb=cod_key.split(',')
            bb1 = {'x1': float(cod_rgb[0]),
               'y1': float(cod_rgb[1]),
               'x2': float(cod_rgb[2]),
               'y2': float(cod_rgb[3])}
            max_iou = 0.0
            overlap_of_key=None
            for cod_key_of in pred_boxes_of:
                cod_of = cod_key_of.split(',')
                bb2 = {'x1': float(cod_of[0]),
                    'y1': float(cod_of[1]),
                    'x2': float(cod_of[2]),
                    'y2': float(cod_of[3])}
                iou=get_iou(bb1,bb2)
                if iou > max_iou:
                    max_iou=iou
                    overlap_of_key=cod_key_of
            # Compare the results from RGB and OF, choose the one with bigger prob
            if overlap_of_key != None:
                if pred_boxes_of[overlap_of_key][0][1] > pred_boxes[cod_key][0][1]:
                    pred_boxes[cod_key][0]=pred_boxes_of[overlap_of_key][0]
        # Visulize the image
        write_vis(pred_boxes,os.path.join(img_path,img_list[t]),os.path.join(out_path,img_list[t]),id2class=id2class)


if __name__ == "__main__":
    main()
