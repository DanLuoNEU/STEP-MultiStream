"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from utils.flow_utils import flow2img

font = ImageFont.truetype("utils/arial.ttf", 28)

def draw_rectangle(draw, coords, outline, width=4):
    for i in range(int(width/2)):
        draw.rectangle(coords-i, outline=outline, fill=None)
    for i in range(1,int(width/2)):
        draw.rectangle(coords+i, outline=outline, fill=None)


def overlay_image(image_path, output_path, gt_boxes=None, pred_boxes=None, id2class=None):
    """
    pred_boxes: a dictionary with boxes as keys and (label, score) as values
    """
    I = Image.open(image_path)
    W, H = I.size

    draw = ImageDraw.Draw(I)
    
    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            draw_rectangle(draw, gt_boxes[i, :4], outline="green")

    if pred_boxes is not None:
        for key in pred_boxes:
            if (len(pred_boxes[key])==1) and (pred_boxes[key][0][0]==3):
                continue
            box = np.asarray(key.split(','), dtype=np.float32)
            box[::2] *= W
            box[1::2] *= H
            draw_rectangle(draw, box, outline="red")

            count = 0
            for label, score in pred_boxes[key]:
                # Dont show background
                if id2class[label]=="background":
                    continue
                draw.text((box[0]+10, box[1]+10+count*20), '{}: {:.2f}'.format(id2class[label] if id2class is not None else label,
                                                                               score), fill="red",font=font)
                break
                count += 1

    I.save(output_path)

def overlay_of(of_path, output_path, gt_boxes=None, pred_boxes=None, id2class=None):
    """
    pred_boxes: a dictionary with boxes as keys and (label, score) as values
    """
    # I = Image.open(image_path)
    I = Image.fromarray(flow2img(np.load(of_path)))
    W, H = I.size

    draw = ImageDraw.Draw(I)
    
    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            draw_rectangle(draw, gt_boxes[i, :4], outline="green")

    if pred_boxes is not None:
        for key in pred_boxes:
            # If there is only one class and the class is 'background'
            if (len(pred_boxes[key])==1) and (pred_boxes[key][0][0]==3):
                continue
            box = np.asarray(key.split(','), dtype=np.float32)
            box[::2] *= W
            box[1::2] *= H
            
            draw_rectangle(draw, box, outline="red")

            count = 0
            for label, score in pred_boxes[key]:
                # Dont show background
                if id2class[label]=="background":
                    continue
                draw.text((box[0]+10, box[1]+10+count*20), '{}: {:.2f}'.format(id2class[label] if id2class is not None else label,
                                                                               score), fill="red",font=font)
                break
                count += 1

    I.save(output_path)