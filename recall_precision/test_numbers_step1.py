# This file is gonna integrated to test_numbers.py at last
# README: if num_classes is changing, modify line 53,68,73
# 20210709, Dan

import os
import sys
import glob
import numpy as np
import pickle

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[0] * inter[1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    np.seterr(divide='ignore', invalid='ignore')
    inter = intersect(box_a, box_b)
    area_a = ((box_a[2]-box_a[0]) *
              (box_a[3]-box_a[1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def main():
    path_label="/data/CLASP-DATA/CLASP2-STEP/data/label_2cls/train.pkl"
    path_results="/home/dan/ws/STEP-MultiStream/test/PVDs-2cls-train_val/Recall_Precision/results_train_conf08.txt"
    path_output="/home/dan/ws/STEP-MultiStream/test/PVDs-2cls-train_val/Recall_Precision/results_train_conf08_FalseAlarm.csv"
    path_vid = "/data/CLASP-DATA/CLASP2-STEP/data/frames"
    label_xfr=2

    with open(path_label,'rb') as fin:
        annots = pickle.load(fin)

    with open(path_results, 'r') as f:
        lines = f.readlines()

    secs=[]
    num_gt = 0
    for name_video in sorted(annots.keys()):
        for fid in annots[name_video].keys():
            for pid in annots[name_video][fid].keys():
                if annots[name_video][fid][pid]['label'] == [label_xfr]:
                    num_gt += 1
    
    records=[]
    num_pos, num_tp = 0, 0
    # print('False Alarms:')
    for id_line,line in enumerate(lines):
        name_video,sid,x0,y0,x1,y1,l,s = line.split(',')
        line_record=name_video+','+sid+','+x0+','+y0+','+x1+','+y1+','+l+','+str(id_line)+'\n'
        sid = int(sid)
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        l = int(l)
        bool_tp=False
        if l == label_xfr: # 2cls
            num_pos += 1
            if name_video in annots.keys():
                if sid in annots[name_video].keys():
                    for pid in annots[name_video][sid].keys():
                        if annots[name_video][sid][pid]['label'] == [label_xfr]:
                            box_gt = annots[name_video][sid][pid]['box']
                            box_res = [x0,y0,x1,y1]
                            res_jar = jaccard_numpy(box_gt, box_res)
                            if res_jar > 0.5:
                                num_tp += 1
                                bool_tp = True
            if not bool_tp:
                records.append(line_record)

    with open(path_output,'w') as f:
        f.writelines(records)
    
    print("Precision: ", num_tp/num_pos, f"{num_tp}/{num_pos}")
    print("Recall: ", num_tp/num_gt, f"{num_tp}/{num_gt}")


    pass

if __name__=="__main__":
    main()